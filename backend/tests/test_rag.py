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
from schemas.rag_schemas import Chunk, RouteDecision
from services import ingest_queue
from services.ingest_service import IngestService
from services.rag_service import RagService, rag_context_tokens


DIM = 8


class FakeEmbed:
    def embed(self, texts):
        return [[float(len(t) % 7)] * DIM for t in texts]

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
        self.assertEqual([h["id"] for h in hits],
                         [chunk_id(uid_b, 0), chunk_id(uid_a, 0)])
        self.assertEqual(hits[0]["text"], "beta bakery")
        self.assertEqual(hits[0]["document_id"], sid(uid_b))

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
        with patch("services.rag_service.get_default_ctx", return_value=2048):
            small = rag_context_tokens()
        with patch("services.rag_service.get_default_ctx", return_value=4096):
            big = rag_context_tokens()
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
            {"document_id": str(uid), "heading": "",
             "page": 2, "text": "x" * 50000, "score": 1.0},
        ]
        with patch("services.rag_service.get_default_ctx", return_value=2048):
            msgs = rag.build_messages("q?", hits, [{"role": "user", "content": "hi"}])
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

    def test_parse_failure_recovers_rag(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("parse failed; raw: RAG")
        out = self.decider(llm).decide("q about docs?", uuid.uuid4())
        self.assertEqual((out.route, out.reason), ("RAG", "recovered"))

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


class QueueMechanics(DbCase):
    def drain(self):
        try:
            while True:
                self.tmp_queue_cleanup.append(ingest_queue._jobs.get_nowait())
                ingest_queue._jobs.task_done()
        except Empty:
            pass

    def setUp(self):
        super().setUp()
        self.tmp_queue_cleanup = []
        self.addCleanup(self.drain)

    def test_submit_rejects_unknown_conversation(self):
        with self.assertRaises(ValueError):
            ingest_queue.submit_ingest(
                self.db, self.write(), "a.md", uuid.uuid4()
            )

    def test_submit_enqueues_without_lock_error(self):
        cid = self.conv()
        with patch.object(ingest_queue, "_ensure_worker", lambda: None):
            d1 = ingest_queue.submit_ingest(self.db, self.write("f1.md"), "f1.md", cid)
            d2 = ingest_queue.submit_ingest(self.db, self.write("f2.md"), "f2.md", cid)
        self.assertEqual(d1.status, "pending")
        self.assertEqual(d2.status, "pending")
        self.assertEqual(ingest_queue._jobs.qsize(), 2)

    def test_superseded_job_is_noop(self):
        cid = self.conv()
        with patch.object(ingest_queue, "SessionLocal", lambda: self.db):
            ingest_queue._process(
                (str(uuid.uuid4()), self.write(), "ghost.md", str(cid), "test-req")
            )  # must not raise

    def test_failed_marker_visible(self):
        cid = self.conv()
        with patch.object(ingest_queue, "SessionLocal", lambda: self.db):
            ingest_queue._mark_failed(
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
        with patch.object(ingest_queue, "_ensure_worker", lambda: None):
            pending = ingest_queue.submit_ingest(
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
        ingest_queue._mark_failed(self.db, temp.id, "a.md", cid, "boom")
        self.assertIsNone(
            self.db.query(DocumentsModel).filter_by(id=temp.id).first())
        kept = self.db.query(DocumentsModel).filter_by(id=live.id).one()
        self.assertEqual((kept.status, kept.filename), ("indexed", "a.md"))

    def test_process_removes_temp_row_on_success(self):
        import services.llm_service as llm_mod
        import services.ingest_service as ingest_mod

        cid = self.conv()
        fake_llm = MagicMock()
        fake_llm.return_value.invoke.return_value = "a short summary"
        canon = Path(self.tmp.name) / "canon.md"
        with patch.object(ingest_queue, "_ensure_worker", lambda: None), \
             patch.object(ingest_queue, "EmbeddingEngine", FakeEmbed), \
             patch.object(ingest_queue, "SessionLocal", lambda: self.db), \
             patch.object(ingest_queue, "LanceRepository",
                          lambda db: self.lance()), \
             patch.object(llm_mod, "LLMService", fake_llm), \
             patch.object(ingest_mod, "stored_upload_path",
                          lambda c, f: canon):
            src = self.write("t.md", "hello world " * 100)
            doc = ingest_queue.submit_ingest(self.db, src, "t.md", cid)
            self.drain()  # keep the real worker out of it
            ingest_queue._process((str(doc.id), src, "t.md", str(cid), "test-req"))
        rows = self.db.query(DocumentsModel).filter(
            DocumentsModel.conversation_id == cid).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            (rows[0].filename, rows[0].status, rows[0].summary),
            ("t.md", "indexed", "a short summary"))

    def test_queue_depth_excludes_self(self):
        cid = self.conv()
        with patch.object(ingest_queue, "_ensure_worker", lambda: None):
            ingest_queue.submit_ingest(self.db, self.write("q1.md"), "q1.md", cid)
            self.assertEqual(ingest_queue.queue_depth(), 0)
            ingest_queue.submit_ingest(self.db, self.write("q2.md"), "q2.md", cid)
            self.assertEqual(ingest_queue.queue_depth(), 1)


class AppContract(unittest.TestCase):
    def test_openapi_builds(self):
        from main import app

        paths = app.openapi()["paths"]
        self.assertIn("/api/chat/stream", paths)
        self.assertIn("/api/ingest", paths)


if __name__ == "__main__":
    unittest.main()
