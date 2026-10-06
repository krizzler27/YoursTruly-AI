"""Model-free RAG tests: stores, chunking, graph, queue mechanics.

Runs offline (no GGUF): memory SQLite + temp LanceDB + fake embedders.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from queue import Empty
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, ConversationsModel, DocumentChunksModel, DocumentsModel
from repository.document_repository import DocumentRepository
from repository.lance_repository import LanceRepository, chunk_id, sid
from schemas.rag_schemas import Chunk, RouteDecision, SearchHit
import services.ingest_job as ingest_job_mod
from services.ingest_service import IngestService
from services.rag_service import RagService, rag_context_tokens


DIM = 8


class FakeEmbed:
    def embed(self, texts):
        return [[float(len(t) % 7)] * DIM for t in texts]

    def unload(self):
        pass

    def is_generating(self):
        return False


class FakeEngineFactory:
    """Stand-in for the engine class: slots resolve to FakeEmbed."""

    @staticmethod
    def get_instance(role="chat", model_path=None):
        return FakeEmbed()


class FakeMemoryEmbed:
    """Deterministic recall vectors: name/address/coffee share one axis."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def embed(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embed down")
        out = []
        for t in texts:
            low = (t or "").lower()
            if "name" in low or "address" in low or "coffee" in low:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    def unload(self):
        pass


class DbCase(unittest.TestCase):
    def setUp(self):
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        self.db.close()

    def lance(self):
        return LanceRepository(self.db, base_dir=Path(self.tmp.name) / "lance")

    def conv(self):
        row = ConversationsModel(title="t")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row.id

    def write(self, name="a.md", text="hello world"):
        p = Path(self.tmp.name) / name
        p.write_text(text, encoding="utf-8")
        return str(p)


class IdFormat(DbCase):
    def test_sid_chunk_id_roundtrip(self):
        uid = uuid.uuid4()
        cid = chunk_id(uid, 3)
        self.assertTrue(cid.startswith(sid(uid)))
        self.assertEqual(cid, f"{uid}:{3}")  # dashed form everywhere

    def test_doc_ids_match_fts_scope(self):
        repo = self.lance()
        uid = uuid.uuid4()
        chunks = [Chunk(text="alpha beta", index=0, vector=[0.1] * DIM)]
        repo.upsert_chunks(uid, chunks, conversation_id=self.conv())
        self.assertEqual(repo.fts_search("alpha", doc_ids=[sid(uid)]), [chunk_id(uid, 0)])
        self.assertEqual(repo.fts_search("alpha", doc_ids=["nope"]), [])


class FtsSafety(DbCase):
    def setUp(self):
        super().setUp()
        self.repo = self.lance()
        uid = uuid.uuid4()
        self.repo.upsert_chunks(
            uid, [Chunk(text="cats and dogs", index=0, vector=[0.1] * DIM)]
        )

    def test_reserved_words_do_not_raise(self):
        for q in ["OR", "AND", "NOT", "or and", "cats OR dogs"]:
            self.assertIsInstance(self.repo.fts_search(q, limit=5), list)

    def test_empty_and_scoped_empty(self):
        self.assertEqual(self.repo.fts_search(""), [])
        self.assertEqual(self.repo.fts_search("cats", doc_ids=[]), [])


class Rrf(DbCase):
    def test_deterministic_merge(self):
        lists = [["x", "y", "z"], ["y", "w"]]
        a = LanceRepository.rrf_scored(lists, top_k=4)
        b = LanceRepository.rrf_scored(lists, top_k=4)
        self.assertEqual(a, b)
        self.assertEqual(a[0][0], "y")  # top-ranked twice wins
        self.assertGreater(a[0][1], a[1][1])
        self.assertEqual(LanceRepository.rrf_scored([[], []]), [])


class HybridVector(DbCase):
    EAST = [1.0] * DIM
    WEST = [-1.0] * DIM

    def _store(self, lance, uid, text, vector, conv):
        lance.upsert_chunks(
            uid, [Chunk(text=text, index=0, vector=vector)],
            conversation_id=conv,
        )
        self.db.add(DocumentChunksModel(
            id=chunk_id(uid, 0), document_id=uid, index=0, text=text))
        self.db.commit()

    def test_hybrid_fuses_vector_and_bm25(self):
        lance = self.lance()
        conv = self.conv()
        uid_a, uid_b = uuid.uuid4(), uuid.uuid4()
        self._store(lance, uid_a, "alpha aerospace", self.EAST, conv)
        self._store(lance, uid_b, "beta bakery", self.WEST, conv)
        # Query vector near A, query words match B: fusion must return both,
        # B first (ranked by both arms beats ranked by one).
        hits = lance.hybrid_search(
            "bakery", list(self.EAST), top_k=5,
            conversation_id=conv, doc_ids=[sid(uid_a), sid(uid_b)])
        self.assertEqual([h.id for h in hits],
                         [chunk_id(uid_b, 0), chunk_id(uid_a, 0)])
        self.assertEqual(hits[0].text, "beta bakery")
        self.assertEqual(hits[0].document_id, sid(uid_b))

    def test_vector_respects_conversation_scope(self):
        lance = self.lance()
        conv_a, conv_b = self.conv(), self.conv()
        uid = uuid.uuid4()
        self._store(lance, uid, "alpha aerospace", self.EAST, conv_a)
        self.assertEqual(
            lance.vector_search(list(self.EAST), conversation_id=conv_b), [])
        self.assertEqual(
            lance.hybrid_search("alpha", list(self.EAST),
                                conversation_id=conv_b, doc_ids=[]), [])


class Chunking(DbCase):
    def svc(self):
        return IngestService(db=self.db, engine=FakeEmbed(), lance=self.lance())

    def test_empty_sources_yield_no_chunks(self):
        s = self.svc()
        self.assertEqual(s.chunk_md(""), [])
        self.assertEqual(s.chunk_txt("  \n "), [])

    def test_txt_and_md_split(self):
        s = self.svc()
        self.assertGreater(len(s.chunk_txt("word " * 2000)), 1)
        chunks = s.chunk_md("# Head\n\n" + "word " * 2000)
        self.assertGreater(len(chunks), 1)
        self.assertIn("Head", chunks[0].heading)

    def test_reject_bad_suffix_and_size(self):
        s = self.svc()
        with self.assertRaises(ValueError):
            s.load_source(self.write("a.exe", "x"))
        big = Path(self.tmp.name) / "big.txt"
        big.write_bytes(b"x" * (26 * 1024 * 1024))
        with self.assertRaises(ValueError):
            s.load_source(str(big))

    def test_ingest_replaces_same_name(self):
        s = self.svc()
        cid = self.conv()
        d1 = s.ingest(self.write("a.md", "# T\n\nversion one"), "a.md", cid)
        d2 = s.ingest(self.write("b.md", "# T\n\nversion two"), "a.md", cid)
        rows = (
            self.db.query(DocumentsModel)
            .filter(DocumentsModel.conversation_id == cid)
            .all()
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].id, d2.id)
        self.assertNotEqual(d1.id, d2.id)
        lance = self.lance()
        self.assertEqual(
            lance.fts_search("version", doc_ids=[sid(d2.id)]),
            [chunk_id(d2.id, 0)],
        )


class Budget(unittest.TestCase):
    def test_derived_not_fixed(self):
        small = rag_context_tokens(2048)
        big = rag_context_tokens(4096)
        self.assertLess(small, big)
        self.assertLessEqual(small + 512 + 750, 2048)  # rag + answer + history fit
        self.assertGreaterEqual(small, 256)

    def test_build_messages_caps_and_cites_page(self):
        db = MagicMock()
        rag = RagService(db=db, engine=MagicMock(), lance=MagicMock(), docs=MagicMock())
        uid = uuid.uuid4()
        db.query.return_value.filter.return_value.all.return_value = [
            MagicMock(id=uid, filename="f.pdf")
        ]
        hits = [
            SearchHit(id=f"{uid}:0", document_id=str(uid), heading="",
                      page=2, text="x" * 50000, score=1.0),
        ]
        msgs = rag.build_messages("q?", hits, [{"role": "user", "content": "hi"}], max_context_tokens=2048)
        system = msgs[0]["content"]
        self.assertIn("[f.pdf:p2]", system)
        self.assertLessEqual(len(system), 2048 * 4 + 8000)
        db.query.return_value.filter.return_value.all.assert_called_once_with()


class Graph(unittest.TestCase):
    def graph(self, rag, decider=None):
        from services.rag_graph import RagGraph

        db = MagicMock()
        llm = MagicMock()
        llm.build_chat_messages.side_effect = lambda h, q: [
            {"role": "user", "content": q}
        ]
        return RagGraph(db=db, rag=rag, decider=decider or MagicMock(), llm=llm)

    def test_rag_route_searches_scoped(self):
        rag = MagicMock()
        rag.search.return_value = [
            {"document_id": "d", "heading": "H", "text": "t", "score": 1.0}
        ]
        rag.build_messages.return_value = [{"role": "system", "content": "g"}]
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="RAG", reason="t")
        g = self.graph(rag, decider)
        cid = uuid.uuid4()
        out = g.run("q", [{"role": "user", "content": "h"}], cid)
        self.assertEqual(out["route"], "RAG")
        _, kwargs = rag.search.call_args
        self.assertEqual(kwargs.get("conversation_id"), cid)
        rag.build_messages.assert_called_once()

    def test_direct_skips_retrieval(self):
        rag = MagicMock()
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        out = self.graph(rag, decider).run("hi")
        self.assertEqual(out["route"], "DIRECT")
        rag.search.assert_not_called()

    def test_retrieval_error_fails_open_direct(self):
        rag = MagicMock()
        rag.search.side_effect = RuntimeError("boom")
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="RAG", reason="t")
        out = self.graph(rag, decider).run("q")
        self.assertEqual(out["route"], "DIRECT")
        self.assertEqual(out["hits"], [])


def _no_laya(test):
    """Pin Laya unavailable so the test exercises the SLM/stub fail-open path.

    The offline suite is model-free: with a backend/laya-model checkout
    present, live Laya would answer first and the SLM contract under test
    would never run.
    """

    def wrapper(self, *args, **kwargs):
        with patch(
            "services.decider.LayaService.get_instance",
            side_effect=RuntimeError("laya down"),
        ):
            return test(self, *args, **kwargs)

    wrapper.__name__ = test.__name__
    return wrapper


class DeciderRecovery(unittest.TestCase):
    def decider(self, llm):
        from services.decider import Decider

        db = MagicMock()
        docs = MagicMock()
        docs.list_by_conversation.return_value = [MagicMock(filename="a.md", summary="s")]
        d = Decider(db=db, llm=llm)
        d.docs = docs
        return d

    def test_recovers_route_from_raw(self):
        from services.decider import _recover_route

        self.assertEqual(_recover_route("parse failed; raw: {\"route\": \"RAG\""), "RAG")
        self.assertEqual(_recover_route("parse failed; raw: go direct please"), "DIRECT")
        self.assertIsNone(_recover_route("parse failed; raw: ???"))

    @_no_laya
    def test_parse_failure_recovers_rag(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("parse failed; raw: RAG")
        out = self.decider(llm).decide("q about docs?", uuid.uuid4())
        self.assertEqual((out.route, out.reason), ("RAG", "recovered"))

    @_no_laya
    def test_parse_failure_without_route_falls_open(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("parse failed; raw: ???")
        out = self.decider(llm).decide("hi", uuid.uuid4())
        self.assertEqual(out.route, "DIRECT")

    def test_empty_query_never_calls_llm(self):
        llm = MagicMock()
        out = self.decider(llm).decide("   ", uuid.uuid4())
        self.assertEqual(out.route, "DIRECT")
        llm.invoke.assert_not_called()


class DeciderHistory(unittest.TestCase):
    def decider(self, llm):
        from services.decider import Decider

        db = MagicMock()
        docs = MagicMock()
        docs.list_by_conversation.return_value = [MagicMock(filename="report.md", summary="s")]
        d = Decider(db=db, llm=llm)
        d.docs = docs
        return d

    def test_followup_without_filename_stays_rag(self):
        llm = MagicMock()
        d = self.decider(llm)
        history = [
            {"role": "user", "content": "what is in report.md?"},
            {"role": "assistant", "content": "it covers sales"},
        ]
        out = d.decide("summarize it", uuid.uuid4(), history=history)
        self.assertEqual(out.route, "RAG")
        llm.invoke.assert_not_called()

    def test_section_followup_stays_rag(self):
        llm = MagicMock()
        d = self.decider(llm)
        history = [
            {"role": "user", "content": "what is in report.md?"},
            {"role": "assistant", "content": "it covers sales"},
        ]
        out = d.decide("what about section 2", uuid.uuid4(), history=history)
        self.assertEqual(out.route, "RAG")
        llm.invoke.assert_not_called()

    @_no_laya
    def test_chitchat_with_inventory_stays_direct(self):
        from services.decider import RouteDecision as RD

        llm = MagicMock()
        llm.invoke.return_value = RD(route="DIRECT", reason="chit-chat")
        d = self.decider(llm)
        history = [
            {"role": "user", "content": "what is in report.md?"},
            {"role": "assistant", "content": "it covers sales"},
        ]
        out = d.decide("how are you today", uuid.uuid4(), history=history)
        self.assertEqual(out.route, "DIRECT")
        llm.invoke.assert_called_once()

    @_no_laya
    def test_needs_memory_stub_keywords(self):
        from services.decider import needs_memory

        self.assertFalse(needs_memory("how are you today"))
        self.assertTrue(needs_memory("please remember my birthday"))
        self.assertTrue(needs_memory("call me Ash"))
        self.assertTrue(needs_memory("my name is Ada"))

    def test_graph_forwards_last_two_history(self):
        from services.rag_graph import RagGraph

        db = MagicMock()
        rag = MagicMock()
        rag.search.return_value = []
        llm = MagicMock()
        llm.build_chat_messages.side_effect = lambda h, q: [
            {"role": "user", "content": q}
        ]
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        g = RagGraph(db=db, rag=rag, decider=decider, llm=llm)
        history = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
            {"role": "user", "content": "three"},
        ]
        g.run("hi", history, uuid.uuid4())
        _, kwargs = decider.decide.call_args
        self.assertEqual(kwargs.get("history"), history[-2:])


class QueueMechanics(DbCase):
    def drain(self):
        try:
            while True:
                self.tmp_queue_cleanup.append(ingest_job_mod._jobs.get_nowait())
                ingest_job_mod._jobs.task_done()
        except Empty:
            pass

    def setUp(self):
        super().setUp()
        self.tmp_queue_cleanup = []
        self.addCleanup(self.drain)

    def test_submit_rejects_unknown_conversation(self):
        with self.assertRaises(ValueError):
            ingest_job_mod.ingest_job.submit(
                self.db, self.write(), "a.md", uuid.uuid4()
            )

    def test_submit_enqueues_without_lock_error(self):
        cid = self.conv()
        with patch.object(ingest_job_mod._runner, "ensure_worker", lambda: None):
            d1 = ingest_job_mod.ingest_job.submit(self.db, self.write("f1.md"), "f1.md", cid)
            d2 = ingest_job_mod.ingest_job.submit(self.db, self.write("f2.md"), "f2.md", cid)
        self.assertEqual(d1.status, "pending")
        self.assertEqual(d2.status, "pending")
        self.assertEqual(ingest_job_mod._jobs.qsize(), 2)

    def test_superseded_job_is_noop(self):
        cid = self.conv()
        with patch.object(ingest_job_mod, "SessionLocal", lambda: self.db):
            ingest_job_mod.ingest_job.run(
                (str(uuid.uuid4()), self.write(), "ghost.md", str(cid), "test-req")
            )  # must not raise

    def test_failed_marker_visible(self):
        cid = self.conv()
        with patch.object(ingest_job_mod, "SessionLocal", lambda: self.db):
            ingest_job_mod.ingest_job._mark_failed(
                self.db, uuid.uuid4(), "bad.md", cid, "nope"
            )
        row = (
            self.db.query(DocumentsModel)
            .filter(DocumentsModel.conversation_id == cid)
            .first()
        )
        self.assertIsNotNone(row)
        self.assertEqual(row.status, "failed")

    def _indexed(self, cid, filename="a.md"):
        row = DocumentsModel(filename=filename, chunk_count=1,
                             status="indexed", conversation_id=cid, summary="s")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def test_submit_keeps_live_index(self):
        cid = self.conv()
        live = self._indexed(cid)
        with patch.object(ingest_job_mod._runner, "ensure_worker", lambda: None):
            pending = ingest_job_mod.ingest_job.submit(
                self.db, self.write("n.md"), "a.md", cid)
        self.assertTrue(pending.filename.startswith("__pending__"))
        self.assertEqual(
            self.db.query(DocumentsModel).filter_by(id=live.id).one().status,
            "indexed")
        visible = DocumentRepository(self.db).list_by_conversation(cid)
        self.assertEqual([d.filename for d in visible], ["a.md"])

    def test_failed_replacement_keeps_live_index(self):
        cid = self.conv()
        live = self._indexed(cid)
        temp = DocumentsModel(filename="__pending__x", chunk_count=0,
                              status="indexing", conversation_id=cid, summary="")
        self.db.add(temp)
        self.db.commit()
        ingest_job_mod.ingest_job._mark_failed(self.db, temp.id, "a.md", cid, "boom")
        self.assertIsNone(
            self.db.query(DocumentsModel).filter_by(id=temp.id).first())
        kept = self.db.query(DocumentsModel).filter_by(id=live.id).one()
        self.assertEqual((kept.status, kept.filename), ("indexed", "a.md"))

    def test_process_removes_temp_row_on_success(self):
        import services.summarize_service as sum_mod
        import services.ingest_service as ingest_mod

        cid = self.conv()
        canon = Path(self.tmp.name) / "canon.md"
        with patch.object(ingest_job_mod._runner, "ensure_worker", lambda: None), \
             patch.object(ingest_job_mod, "EmbeddingEngine", FakeEngineFactory), \
             patch.object(ingest_job_mod, "SessionLocal", lambda: self.db), \
             patch.object(ingest_job_mod, "LanceRepository",
                          lambda db: self.lance()), \
             patch.object(sum_mod, "summarize_text",
                          return_value="a short summary"), \
             patch.object(ingest_mod, "stored_upload_path",
                          lambda c, f: canon):
            src = self.write("t.md", "hello world " * 100)
            doc = ingest_job_mod.ingest_job.submit(self.db, src, "t.md", cid)
            self.drain()  # keep the real worker out of it
            ingest_job_mod.ingest_job.run((str(doc.id), src, "t.md", str(cid), "test-req"))
        rows = self.db.query(DocumentsModel).filter(
            DocumentsModel.conversation_id == cid).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (rows[0].filename, rows[0].status, rows[0].summary),
            ("t.md", "indexed", "a short summary"))

    def test_submit_enqueues_two_jobs(self):
        cid = self.conv()
        with patch.object(ingest_job_mod._runner, "ensure_worker", lambda: None):
            ingest_job_mod.ingest_job.submit(self.db, self.write("q1.md"), "q1.md", cid)
            ingest_job_mod.ingest_job.submit(self.db, self.write("q2.md"), "q2.md", cid)
            self.assertEqual(ingest_job_mod._jobs.qsize(), 2)


class SemanticMemory(DbCase):
    def repo(self):
        from repository.semantic_repository import SemanticMemoryRepository

        return SemanticMemoryRepository(self.db)

    def test_upsert_and_recall(self):
        r = self.repo()
        row = r.upsert("name", "Ada")
        self.assertEqual(row.value, "Ada")
        self.assertEqual(r.get_by_key("name").value, "Ada")
        r.upsert("name", "Grace")
        self.assertEqual(r.get_by_key("NAME").value, "Grace")
        self.assertEqual(len(r.list_all(limit=100)), 1)

    def test_find_relevant_key_overlap(self):
        r = self.repo()
        r.upsert("name", "Ada")
        r.upsert("favorite food", "ramen")
        hits = r.find_relevant("what is my name?", limit=5)
        self.assertEqual([h.key for h in hits], ["name"])
        self.assertEqual(r.find_relevant("how are you today?", limit=5), [])

    def test_delete_by_key(self):
        r = self.repo()
        r.upsert("name", "Ada")
        self.assertTrue(r.delete("name"))
        self.assertIsNone(r.get_by_key("name"))
        self.assertFalse(r.delete("name"))

    @_no_laya
    def test_needs_memory_true_false(self):
        from types import SimpleNamespace

        from services.decider import needs_memory

        self.assertFalse(needs_memory("how are you today"))
        self.assertTrue(needs_memory("please remember my birthday"))
        self.assertTrue(
            needs_memory(
                "anything else?",
                history=[{"role": "user", "content": "i like strong coffee"}],
            )
        )
        self.assertTrue(
            needs_memory(
                "what is my name?",
                memories=[SimpleNamespace(key="name", value="Ada")],
            )
        )
        self.assertTrue(
            needs_memory(
                "what is my name?", memories=[{"key": "name", "value": "Ada"}]
            )
        )
        self.assertFalse(
            needs_memory(
                "how are you?", memories=[{"key": "name", "value": "Ada"}]
            )
        )

    def test_build_messages_includes_memory_within_cap(self):
        from core.context_budget import allocate, count_tokens, truncate_text

        rag = RagService(
            db=self.db, engine=MagicMock(), lance=MagicMock(), docs=MagicMock()
        )
        big = "name: " + "Ada " * 5000
        msgs = rag.build_messages("what is my name?", None, [], memory_text=big)
        system = msgs[0]["content"]
        budget = allocate(
            route="DIRECT",
            needs_memory=True,
            query_tokens=count_tokens("what is my name?"),
        )
        expect = truncate_text(big.strip(), budget["mem_cap"])
        self.assertIn(expect, system)
        self.assertLessEqual(count_tokens(expect), budget["mem_cap"])
        self.assertNotIn(big.strip(), system)

    def test_build_messages_rag_includes_memory(self):
        rag = RagService(
            db=self.db, engine=MagicMock(), lance=MagicMock(), docs=MagicMock()
        )
        uid = uuid.uuid4()
        hits = [
            SearchHit(id=f"{uid}:0", document_id=str(uid), heading="H",
                      page=None, text="grounded fact", score=1.0),
        ]
        msgs = rag.build_messages(
            "what is my name?", hits, [], memory_text="name: Ada"
        )
        self.assertIn("Ada", msgs[0]["content"])
        self.assertIn("grounded fact", msgs[0]["content"])

    @_no_laya
    def test_memory_error_fails_open(self):
        from services.rag_graph import RagGraph

        db = MagicMock()
        rag = MagicMock()
        rag.db = MagicMock()
        rag.build_messages.return_value = [{"role": "user", "content": "hi"}]
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        with patch("services.rag_graph.SemanticMemoryRepository") as repo:
            repo.return_value.find_relevant.side_effect = RuntimeError("db down")
            out = RagGraph(db=db, rag=rag, decider=decider, llm=MagicMock()).run("hi")
        self.assertEqual(out["route"], "DIRECT")
        _, kwargs = rag.build_messages.call_args
        self.assertIsNone(kwargs.get("memory_text"))

    def test_find_relevant_vector_rescues_paraphrase(self):
        r = self.repo()
        r.upsert("name", "Ada")
        r.upsert("favorite food", "ramen")
        q = "how should I address you"
        self.assertEqual(r.find_relevant(q, limit=5), [])
        hits = r.find_relevant(
            q, limit=5,
            query_vector=[1.0, 0.0],
            candidate_vectors={"name": [1.0, 0.0], "favorite food": [0.0, 1.0]},
        )
        self.assertEqual([h.key for h in hits], ["name"])

    def test_find_relevant_exact_match_stays_first(self):
        r = self.repo()
        r.upsert("name", "Ada")
        r.upsert("favorite food", "ramen")
        hits = r.find_relevant(
            "what is my name?", limit=5,
            query_vector=[0.0, 1.0],
            candidate_vectors={"name": [1.0, 0.0], "favorite food": [0.0, 1.0]},
        )
        self.assertEqual([h.key for h in hits], ["name", "favorite food"])

    def test_find_relevant_bad_vectors_stay_overlap_only(self):
        r = self.repo()
        r.upsert("name", "Ada")
        r.upsert("favorite food", "ramen")
        q = "what is my name?"
        dim_mismatch = r.find_relevant(
            q, limit=5,
            query_vector=[1.0],
            candidate_vectors={"name": [1.0, 0.0], "favorite food": [0.0, 1.0]},
        )
        self.assertEqual([h.key for h in dim_mismatch], ["name"])
        no_candidates = r.find_relevant(q, limit=5, query_vector=[1.0, 0.0])
        self.assertEqual([h.key for h in no_candidates], ["name"])

    def memory_graph(self, engine):
        from services.rag_graph import RagGraph

        rag = MagicMock()
        rag.db = self.db
        rag.engine = engine
        rag.build_messages.side_effect = lambda q, h, hist, **kw: [
            {"role": "u", "content": kw.get("memory_text") or ""}
        ]
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        return RagGraph(db=self.db, rag=rag, decider=decider, llm=MagicMock()), rag

    @_no_laya
    def test_memory_text_single_embed_covers_both_pools(self):
        from repository.episodic_repository import EpisodicMemoryRepository

        self.repo().upsert("name", "Ada")
        self.repo().upsert("favorite food", "ramen")
        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(
            cid, "user prefers strong coffee", 0, 4
        )
        engine = FakeMemoryEmbed()
        g, _ = self.memory_graph(engine)
        text = g._memory_text("how should I address you", cid)
        self.assertIn("name: Ada", text)
        self.assertIn("earlier: user prefers strong coffee", text)
        self.assertNotIn("ramen", text)
        self.assertEqual(len(engine.calls), 1)  # query plus candidates, one call
        self.assertEqual(len(engine.calls[0]), 4)

    @_no_laya
    def test_memory_text_embed_failure_falls_back_overlap_only(self):
        self.repo().upsert("name", "Ada")
        engine = FakeMemoryEmbed(fail=True)
        g, _ = self.memory_graph(engine)
        self.assertIn("Ada", g._memory_text("what is my name?") or "")
        self.assertIsNone(g._memory_text("how should I address you"))
        self.assertEqual(len(engine.calls), 2)  # tried once per turn, then fallback

    @_no_laya
    def test_memory_less_direct_turn_never_touches_engine(self):
        engine = FakeMemoryEmbed()
        g, rag = self.memory_graph(engine)
        out = g.run("how are you today")
        self.assertEqual(out["route"], "DIRECT")
        rag.search.assert_not_called()
        self.assertEqual(engine.calls, [])
        _, kwargs = rag.build_messages.call_args
        self.assertIsNone(kwargs.get("memory_text"))

    @_no_laya
    def test_gate_triple_back_compat(self):
        from services.decider import memory_evidence, needs_memory

        self.assertFalse(needs_memory("how are you today"))
        self.assertTrue(needs_memory("please remember my birthday"))
        self.assertEqual(
            tuple(memory_evidence("how are you today")), (False, False, False)
        )
        self.assertEqual(
            tuple(memory_evidence("please remember my birthday")),
            (True, True, False),
        )

    @_no_laya
    def test_gate_episodic_overlap(self):
        from repository.episodic_repository import EpisodicMemoryRepository
        from services.decider import memory_evidence

        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(
            cid, "we decide the launch plan last tuesday", 0, 4
        )
        self.assertEqual(
            tuple(
                memory_evidence(
                    "what did we decide last tuesday",
                    db=self.db,
                    conversation_id=cid,
                )
            ),
            (True, False, True),
        )
        rows = EpisodicMemoryRepository(self.db).list_recent_for_query(cid, limit=10)
        self.assertEqual(
            tuple(memory_evidence("what did we decide last tuesday", epi_rows=rows)),
            (True, False, True),
        )

    def test_combine_episodic_heavy_gives_epi_most_of_cap(self):
        from services.rag_graph import _combine_memory

        sem = "name: " + "Ada " * 100
        epi = "we decide the launch plan last tuesday with the team"
        out = _combine_memory(sem, [epi], sem_hit=False, epi_hit=True, cap=40)
        self.assertIn("earlier: " + epi, out)
        self.assertNotIn(sem.strip(), out)

    def test_combine_semantic_heavy_gives_sem_most_of_cap(self):
        from services.rag_graph import _combine_memory

        sem = "name: " + "Ada " * 100
        epi = "we decide the launch plan last tuesday with the team"
        out = _combine_memory(sem, [epi], sem_hit=True, epi_hit=False, cap=40)
        self.assertIn("name: Ada", out)
        self.assertNotIn("earlier: " + epi, out)

    def test_combine_no_evidence_keeps_semantic_first(self):
        from services.rag_graph import _combine_memory

        big_sem = "name: " + "Ada " * 5000
        out = _combine_memory(big_sem, ["user likes ramen"])
        self.assertIn("name:", out)
        self.assertNotIn("ramen", out)

    @_no_laya
    def test_memory_text_episodic_error_degrades_to_semantic_only(self):
        from repository.episodic_repository import EpisodicMemoryRepository

        self.repo().upsert("name", "Ada")
        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(cid, "user likes ramen", 0, 8)
        engine = FakeMemoryEmbed()
        g, _ = self.memory_graph(engine)
        with patch.object(
            EpisodicMemoryRepository,
            "list_recent_for_query",
            side_effect=RuntimeError("db down"),
        ):
            text = g._memory_text("what is my name?", cid)
        self.assertIn("Ada", text or "")
        self.assertNotIn("earlier:", text or "")

    @_no_laya
    def test_memory_text_episodic_heavy_threading(self):
        from unittest.mock import PropertyMock

        from core.context_budget import count_tokens
        from repository.episodic_repository import EpisodicMemoryRepository
        from services.decider import memory_evidence

        self.repo().upsert("pet", "Bex")
        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(
            cid, "we decide the launch plan last tuesday", 0, 4
        )
        self.assertEqual(
            tuple(
                memory_evidence(
                    "what did we decide last tuesday",
                    db=self.db,
                    conversation_id=cid,
                )
            ),
            (True, False, True),
        )
        engine = FakeMemoryEmbed()
        g, _ = self.memory_graph(engine)
        with patch(
            "config.Config.EFFECTIVE_MEMORY_TOKENS",
            new_callable=PropertyMock,
            return_value=60,
        ):
            text = g._memory_text("what did we decide last tuesday", cid)
        self.assertIn("tuesday", text or "")
        self.assertIn("Bex", text or "")
        self.assertLessEqual(count_tokens(text or ""), 60)


class TopicBuffer(DbCase):
    def chat_repo(self):
        from repository.chat_repository import ChatRepository

        return ChatRepository(self.db)

    def epi_repo(self):
        from repository.episodic_repository import EpisodicMemoryRepository

        return EpisodicMemoryRepository(self.db)

    def tag(self, cid, topic):
        return self.chat_repo().set_topic(cid, topic)

    def tgraph(self):
        from services.rag_graph import RagGraph

        rag = RagService(
            db=self.db, engine=FakeEmbed(), lance=self.lance(), docs=MagicMock()
        )
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        return RagGraph(db=self.db, rag=rag, decider=decider, llm=MagicMock())

    def test_set_clear_list(self):
        repo = self.chat_repo()
        c1, c2, c3 = self.conv(), self.conv(), self.conv()
        self.tag(c1, "laya")
        self.tag(c2, "laya")
        self.tag(c3, "other")
        self.assertEqual(repo.get_by_id(c1).topic, "laya")
        self.assertEqual(repo.get_by_id(c3).topic, "other")
        got = repo.list_by_topic("laya", limit=10, exclude_id=c1)
        self.assertEqual([r.id for r in got], [c2])
        self.assertEqual(repo.list_by_topic("  ", limit=10), [])
        self.assertEqual(repo.list_by_topic("missing", limit=10), [])
        self.tag(c1, None)
        self.assertIsNone(repo.get_by_id(c1).topic)
        self.tag(c2, "   ")
        self.assertIsNone(repo.get_by_id(c2).topic)
        self.assertIsNone(repo.set_topic(uuid.uuid4(), "laya"))

    @_no_laya
    def test_sibling_lines_appear_with_label(self):
        c1, c2 = self.conv(), self.conv()
        self.tag(c1, "laya")
        self.tag(c2, "laya")
        self.epi_repo().create(c2, "decided the launch date is friday", 0, 4)
        g = self.tgraph()
        lines = g._topic_sibling_lines(c1)
        self.assertEqual(len(lines), 1)
        self.assertIn("Earlier in project laya:", lines[0])
        self.assertIn("friday", lines[0])
        out = g.run("what did we decide?", [], c1)
        self.assertEqual(out["route"], "DIRECT")
        system = out["messages"][0]["content"]
        self.assertIn("Earlier in project laya:", system)
        self.assertIn("friday", system)

    @_no_laya
    def test_cap_respected_siblings_yield_to_own_history(self):
        from core.context_budget import allocate as real_allocate

        c1, c2 = self.conv(), self.conv()
        self.tag(c1, "laya")
        self.tag(c2, "laya")
        self.epi_repo().create(c2, "sibling decision about friday", 0, 4)
        history = [
            {"role": "user", "content": "alpha " * 40},
            {"role": "assistant", "content": "beta " * 40},
        ]
        g = self.tgraph()
        out = g.run("what did we decide?", history, c1)
        self.assertIn("Earlier in project laya:", out["messages"][0]["content"])

        def tiny_allocate(*a, **k):
            budget = real_allocate(*a, **k)
            budget["history_cap"] = 4
            return budget

        with patch("services.rag_service.allocate", side_effect=tiny_allocate):
            starved = g.run("what did we decide?", history, c1)
        system = starved["messages"][0]["content"]
        self.assertNotIn("Earlier in project", system)
        self.assertIn("beta", system)  # own last turn survives the squeeze

    @_no_laya
    def test_other_topic_excluded(self):
        c1, c2 = self.conv(), self.conv()
        self.tag(c1, "laya")
        self.tag(c2, "other")
        self.epi_repo().create(c2, "unrelated decision about monday", 0, 4)
        g = self.tgraph()
        self.assertEqual(g._topic_sibling_lines(c1), [])
        out = g.run("what did we decide?", [], c1)
        self.assertNotIn("Earlier in project", out["messages"][0]["content"])
        self.assertNotIn("monday", out["messages"][0]["content"])

    @_no_laya
    def test_untagged_unchanged(self):
        c1, c2 = self.conv(), self.conv()
        self.tag(c2, "laya")
        self.epi_repo().create(c2, "sibling decision about friday", 0, 4)
        g = self.tgraph()
        self.assertEqual(g._topic_sibling_lines(c1), [])
        self.assertEqual(g._topic_sibling_lines(None), [])
        out = g.run("hello there", [], c1)
        self.assertNotIn("Earlier in project", out["messages"][0]["content"])
        rag = g.rag
        plain = rag.build_messages("hello there", None, [])
        empty = rag.build_messages("hello there", None, [], topic_lines=[])
        self.assertEqual(plain, empty)

    @_no_laya
    def test_max_three_lines(self):
        c1 = self.conv()
        self.tag(c1, "laya")
        for i in range(5):
            sib = self.conv()
            self.tag(sib, "laya")
            self.epi_repo().create(sib, f"sibling summary number {i}", 0, 4)
        lines = self.tgraph()._topic_sibling_lines(c1)
        self.assertEqual(len(lines), 3)
        self.assertTrue(all("Earlier in project laya:" in line for line in lines))

    def test_api_patch_topic_round_trip(self):
        from pydantic import ValidationError

        from schemas.api_schemas import ConversationResponse, ConversationUpdateRequest
        from services.chat_services import ChatServices

        req = ConversationUpdateRequest.model_validate({"topic": "laya"})
        self.assertEqual((req.title, req.topic), (None, "laya"))
        self.assertIn("topic", req.model_fields_set)
        blank = ConversationUpdateRequest.model_validate({"topic": "   "})
        self.assertIsNone(blank.topic)
        untouched = ConversationUpdateRequest.model_validate({"title": "Keep"})
        self.assertNotIn("topic", untouched.model_fields_set)
        with self.assertRaises(ValidationError):
            ConversationUpdateRequest.model_validate({"topic": "x" * 65})

        svc = ChatServices(self.db)
        conv = svc.ensure_conversation(None, title="Original")
        self.assertIsNone(conv.topic)
        svc.set_topic(conv.id, "laya")
        self.assertEqual(svc.chat_repo.get_by_id(conv.id).topic, "laya")
        resp = ConversationResponse.model_validate(svc.chat_repo.get_by_id(conv.id))
        self.assertEqual(resp.topic, "laya")
        svc.set_topic(conv.id, "")
        self.assertIsNone(svc.chat_repo.get_by_id(conv.id).topic)

        from router.chat_api import update_conversation

        tagged = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"topic": "laya"}), self.db
        )
        self.assertEqual(tagged.topic, "laya")
        self.assertEqual(tagged.title, "Original")  # topic-only patch keeps title
        renamed = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"title": "New name"}), self.db
        )
        self.assertEqual(renamed.title, "New name")
        self.assertEqual(renamed.topic, "laya")  # title-only patch keeps topic
        cleared = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"topic": None}), self.db
        )
        self.assertIsNone(cleared.topic)


class AppContract(unittest.TestCase):
    def test_openapi_builds(self):
        from main import app

        paths = app.openapi()["paths"]
        self.assertIn("/api/chat/stream", paths)
        self.assertIn("/api/ingest", paths)


if __name__ == "__main__":
    unittest.main()
