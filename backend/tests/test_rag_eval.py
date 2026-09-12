"""Whole-RAG eval on 10 sampled Qs through prod code paths only.

Per doc: bytes -> staged_upload_path -> submit_ingest -> real queue
worker (summaries included) -> poll until indexed.
Per question: RagGraph.run (real SLM gate, real search, real build)
-> live generation with prod params (temp by route, repeat 1.1).

Isolation: temp file SQLite + temp LanceDB + one random conversation
dir under the real docs root (removed at teardown). Two documented
seams (worker import-time globals that must point at temp):
ingest_queue.SessionLocal, ingest_queue.LanceRepository.
LLAMA_MODEL_PATH stays real so engines resolve their GGUFs.
All logic executed is unmodified prod code.

LangSmith gate: tracing must be on (env), else the run asks whether
to continue without traces. Secrets are never read or printed here;
load_dotenv only mirrors what the app itself does at startup.

Slow (~15-25 min on CPU): run explicitly, not with the fast suite:
    .venv\\Scripts\\python.exe -m unittest tests.test_rag_eval -v
"""

import json
import os
import sys
import tempfile
import time
import unittest
import uuid
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, ConversationsModel, DocumentsModel
from repository.lance_repository import LanceRepository as RealLance
from services import ingest_queue
from services.ingest_service import staged_upload_path, stored_upload_path
from services.llama_engine import EmbeddingEngine
from services.rag_graph import RagGraph
from services.rag_service import RagService

BACKEND = Path(__file__).resolve().parent.parent
DOCS = BACKEND / "rag_eval" / "docs"
QA = BACKEND / "rag_eval" / "qa.jsonl"
TRACE = BACKEND / "rag_eval" / "eval-trace.jsonl"  # git-ignored

N_QS = 10
POLL_S = 3
INGEST_DEADLINE_S = 1500


def ensure_tracing():
    """LangSmith on, or owner explicitly accepts a tracelsess run."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BACKEND / ".env")
    except Exception:
        pass
    on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.environ.get("LANGSMITH_API_KEY")
    )
    if on:
        print("LangSmith tracing ON (LangGraph runs will be traced)")
        return
    ans = input(
        "LangSmith tracing is OFF (need LANGSMITH_TRACING=true + "
        "LANGSMITH_API_KEY in env). Continue without traces? [y/N] "
    ).strip().lower()
    if ans != "y":
        raise unittest.SkipTest("owner declined run without LangSmith tracing")


def sample_rows(n=N_QS):
    """Round-robin across sources so all doc types (pdf/md/txt) are hit."""
    rows = [json.loads(line) for line in QA.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r.get("expected_source")]  # route-only rows skipped
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["expected_source"]].append(r)
    out, i = [], 0
    while len(out) < n:
        progressed = False
        for source in sorted(by_source):
            if i < len(by_source[source]) and len(out) < n:
                out.append(by_source[source][i])
                progressed = True
        i += 1
        if not progressed:
            break
    return out


class WholeRagEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_tracing()

        # Temp FILE db, not :memory:: the queue worker runs on its own
        # thread and each thread gets a private :memory: database.
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.db_path = str(Path(cls.tmp.name) / "eval.db")
        cls.eng = create_engine(f"sqlite:///{cls.db_path}")
        Base.metadata.create_all(cls.eng)
        cls.TestSessions = sessionmaker(bind=cls.eng)
        cls.db = cls.TestSessions()
        # LIFO cleanups: session + pooled handles released before rmtree
        # (Windows cannot delete an open SQLite file).
        cls.addClassCleanup(cls.eng.dispose)
        cls.addClassCleanup(cls.db.close)

        root = Path(cls.tmp.name)
        cls.lance_dir = root / "lance"

        # Temp seams: worker internals must use the test DB and temp
        # vector store. LLAMA_MODEL_PATH stays real so the embedder and
        # chat engine resolve their GGUFs; eval files live under one
        # random conversation dir, removed at teardown.
        cls._patches = [
            patch.object(ingest_queue, "SessionLocal", cls.TestSessions),
            patch.object(
                ingest_queue, "LanceRepository",
                lambda db: RealLance(db, base_dir=cls.lance_dir),
            ),
        ]
        for p in cls._patches:
            p.start()
        cls.addClassCleanup(lambda: [p.stop() for p in cls._patches])
        cls.embed = EmbeddingEngine()
        cls.addClassCleanup(cls.embed.unload)

        cls.conv = uuid.uuid4()
        cls.db.add(ConversationsModel(id=cls.conv, title="eval"))
        cls.db.commit()

        import shutil

        cls.addClassCleanup(
            lambda: shutil.rmtree(
                stored_upload_path(cls.conv, "x").parent, ignore_errors=True
            )
        )

        cls.qs = sample_rows()
        sources = sorted({q["expected_source"] for q in cls.qs})
        print(f"\neval: {len(cls.qs)} Qs over {sources}")
        cls._ingest_all(sources)
        cls._assert_summaries(sources)

        cls.rag = RagService(
            db=cls.db,
            engine=cls.embed,
            lance=RealLance(cls.db, base_dir=cls.lance_dir),
        )
        cls.graph = RagGraph(db=cls.db, rag=cls.rag)
        cls.trace_lines = []

    @classmethod
    def _ingest_all(cls, sources):
        """Prod ingest path per file, then poll like a client would."""
        for name in sources:
            src = DOCS / name
            assert src.exists(), f"missing doc: {src}"
            staged = staged_upload_path(cls.conv, name)
            staged.write_bytes(src.read_bytes())
            ingest_queue.submit_ingest(cls.db, str(staged), name, cls.conv)

        deadline = time.time() + INGEST_DEADLINE_S
        while True:
            cls.db.expire_all()
            rows = (
                cls.db.query(DocumentsModel)
                .filter(DocumentsModel.conversation_id == cls.conv)
                .all()
            )
            by_status = defaultdict(list)
            for r in rows:
                by_status[r.status].append(r.filename)
            pending = by_status["pending"] + by_status["indexing"]
            failed = [r for r in rows if r.status == "failed"]
            if failed:
                detail = "; ".join(f"{r.filename}: {r.summary}" for r in failed)
                raise AssertionError(f"ingest failed: {detail}")
            # Indexed row commits before its summary does; wait for both.
            unsummarized = [r.filename for r in rows
                            if not (r.summary or "").strip()]
            if not pending and len(rows) == len(sources) and not unsummarized:
                for r in rows:
                    print(f"ingested {r.filename}: "
                          f"{r.chunk_count} chunks")
                return
            if time.time() > deadline:
                raise AssertionError(
                    f"ingest timed out, pending={pending} "
                    f"unsummarized={unsummarized}")
            time.sleep(POLL_S)

    @classmethod
    def _assert_summaries(cls, sources):
        """The gap that once blinded the gate: summaries must exist."""
        rows = (
            cls.db.query(DocumentsModel)
            .filter(DocumentsModel.conversation_id == cls.conv)
            .all()
        )
        missing = [r.filename for r in rows if not (r.summary or "").strip()]
        self_assert = (
            f"summaries missing for {missing}: gate would see bare filenames"
        )
        assert not missing, self_assert
        print(f"summaries present for all {len(sources)} docs")

    def _trace(self, entry):
        type(self).trace_lines.append(entry)
        with open(TRACE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _names(self, hits):
        names = []
        for h in hits:
            row = self.rag.docs.get_by_id(uuid.UUID(h["document_id"]))
            names.append(row.filename if row else "?")
        return names

    def test_1_recall_and_grounding(self):
        gated_rag = r1 = r5 = cited = 0
        type(self).results = {}
        for q in self.qs:
            with self.subTest(id=q["id"], query=q["query"][:60]):
                t0 = time.perf_counter()
                out = self.graph.run(
                    q["query"], [{"role": "user", "content": "hi"}], self.conv)
                gate_ms = (time.perf_counter() - t0) * 1000
                route, hits = out["route"], out.get("hits", [])
                reason = (out.get("decision").reason
                          if out.get("decision") else "?")
                names = self._names(hits) if hits else []
                scores = [round(h.get("score", 0.0), 4) for h in hits]
                print(f"\nQ{q['id']} route={route} reason={reason} "
                      f"gate={gate_ms:.0f}ms stages={out.get('stages')} "
                      f"hits={list(zip(names, scores))}")
                self._trace({"id": q["id"], "query": q["query"], "route": route,
                             "reason": reason, "gate_ms": round(gate_ms),
                             "stages": out.get("stages"), "hits": names,
                             "scores": scores,
                             "expected_source": q["expected_source"]})
                if route != "RAG":
                    continue  # gate signal, reported below, not asserted
                gated_rag += 1
                self.assertTrue(hits, "RAG route but no hits")
                if names[0] == q["expected_source"]:
                    r1 += 1
                self.assertIn(q["expected_source"], names,
                              f"expected {q['expected_source']} in {names}")
                r5 += 1
                # Heading-less chunks cite as [file], headed as [file:...].
                self.assertTrue(
                    f"[{q['expected_source']}" in out["messages"][0]["content"],
                    f"no citation for {q['expected_source']}")
                cited += 1
                type(self).results[q["id"]] = (q, out)
        print(f"\neval result: gate=RAG {gated_rag}/{len(self.qs)} "
              f"recall@1={r1}/{gated_rag} recall@5={r5}/{gated_rag} "
              f"cited={cited}/{gated_rag} (trace: {TRACE.name})")

    def test_2_generation(self):
        from services.llm_service import LLMService

        self.assertTrue(getattr(type(self), "results", None),
                        "recall test must run first")
        # One model resident at a time on 8GB: embedder's job is done.
        self.embed.unload()
        llm = LLMService()
        sub_hit = answered = 0
        for qid, (q, out) in type(self).results.items():
            with self.subTest(id=qid, query=q["query"][:60]):
                route = out["route"]
                answer = llm.invoke(
                    out["messages"],
                    max_tokens=256,
                    temperature=0.5 if route == "RAG" else 0.6,
                    repeat_penalty=1.1,  # prod parity with chat_api
                )
                self.assertTrue((answer or "").strip(), "empty generation")
                answered += 1
                norm = lambda s: " ".join((s or "").lower().split())
                hit = norm(q.get("expected_answer", "")) and \
                    norm(q["expected_answer"]) in norm(answer)
                sub_hit += bool(hit)
                print(f"\nQ{qid} [{'SUB-HIT' if hit else 'sub-miss'}] "
                      f"{q['query'][:70]}\nA: {answer[:300]}")
                self._trace({"id": qid, "answer_head": answer[:500],
                             "sub_hit": bool(hit),
                             "expected_answer": q.get("expected_answer")})
        print(f"\ngeneration: answered={answered}/{len(type(self).results)} "
              f"expected-substring={sub_hit}/{len(type(self).results)} "
              f"(informational: paraphrase still counts as correct)")


if __name__ == "__main__":
    unittest.main()
