"""Laya L1 tests: gating policy with a faked agent, no weights.

Covers the decider swap (route plus needs-memory in one call, threshold
gating, safe-defaults, SLM fallback) and the LayaService lifecycle
(singleton, transient load/unload, clear load errors). Never touches the
real backend/laya-model weights.
"""

import sys
import asyncio
import json
import unittest
import uuid
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.rag_schemas import RouteDecision, SearchHit
from services import decider as dec_mod
from services import rag_service as rag_mod
from services.decider import Decider, _laya_route, needs_memory
from services.laya_service import (
    HIT_GRADE_QUESTION,
    NEEDS_MEMORY_QUESTION,
    ROUTE_QUESTION,
    LayaService,
)


class FakeAgent:
    """Stand-in for laya.Agent: canned answers or a load-time error."""

    def __init__(self, answers=None, error=None):
        self.answers = answers or {}
        self.error = error
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        return {"answers": self.answers}


def _route_answers(label, conf, p_mem):
    return {
        "route": {"choice": label, "confidence": conf},
        "needs_memory": {"noul": p_mem},
    }


class LayaDecider(unittest.TestCase):
    def tearDown(self):
        LayaService.reset_instance()

    def decider(self, laya_answers=None, laya_error=None, llm_result=None):
        db = MagicMock()
        llm = MagicMock()
        if llm_result is not None:
            llm.invoke.return_value = llm_result
        d = Decider(db=db, llm=llm)
        docs = MagicMock()
        doc = MagicMock(filename="a.md", summary="s")
        docs.list_by_conversation.return_value = [doc]
        docs.list_recent.return_value = [doc]
        d.docs = docs
        svc = LayaService(model_dir="fake-dir")
        fake = FakeAgent(answers=laya_answers, error=laya_error)
        svc._agent = fake
        inst = patch.object(dec_mod.LayaService, "get_instance", return_value=svc)
        inst.start()
        self.addCleanup(inst.stop)
        repo = patch.object(dec_mod, "SemanticMemoryRepository")
        repo.start()
        self.addCleanup(repo.stop)
        dec_mod.SemanticMemoryRepository.return_value.find_relevant.return_value = []
        self._svc = svc
        self._fake = fake
        return d, llm

    def test_threshold_accept_uses_laya_label(self):
        d, llm = self.decider(_route_answers("RAG", 0.85, 0.9))
        out = d.decide(
            "where is the nightly backup timetable written down?", uuid.uuid4()
        )
        self.assertEqual((out.route, out.reason), ("RAG", "laya"))
        llm.invoke.assert_not_called()
        state, questions = self._fake.calls[0]
        self.assertEqual(
            set(questions), {"route", "needs_memory"}
        )  # one combined call
        self.assertEqual(
            set(state), {"query", "history", "attached_docs"}
        )
        self.assertIn("a.md", state["attached_docs"])

    def test_below_threshold_safe_default_rag_with_docs(self):
        d, llm = self.decider(_route_answers("DIRECT", 0.5, 0.1))
        out = d.decide("where is the backup timetable?", uuid.uuid4())
        self.assertEqual(out.route, "RAG")
        self.assertEqual(out.reason, "laya low-conf safe-default")
        llm.invoke.assert_not_called()

    def test_policy_direct_without_docs(self):
        out = _laya_route("RAG", 0.5, False, 0.8)
        self.assertEqual((out.route, out.reason), ("DIRECT", "laya low-conf no docs"))

    def test_policy_accept_and_web_fallback(self):
        out = _laya_route("DIRECT", 0.9, True, 0.8)
        self.assertEqual((out.route, out.reason), ("DIRECT", "laya"))
        out = _laya_route("WEB", 0.95, True, 0.8)
        self.assertEqual(out.route, "RAG")  # no WEB branch yet, safe-default

    def test_laya_error_falls_back_to_slm_path(self):
        d, llm = self.decider(
            laya_error=RuntimeError("weights gone"),
            llm_result=RouteDecision(route="DIRECT", reason="chit-chat"),
        )
        out = d.decide("how are you today", uuid.uuid4())
        self.assertEqual((out.route, out.reason), ("DIRECT", "chit-chat"))
        llm.invoke.assert_called_once()  # today's SLM path exactly

    def test_needs_memory_laya_first(self):
        d, _ = self.decider({"needs_memory": {"noul": 0.95}})
        self.assertTrue(needs_memory("how are you today"))

    def test_needs_memory_stub_below_threshold(self):
        d, _ = self.decider({"needs_memory": {"noul": 0.1}})
        self.assertTrue(needs_memory("please remember my birthday"))
        self.assertFalse(needs_memory("how are you today"))

    def test_needs_memory_stub_on_laya_error(self):
        d, _ = self.decider(laya_error=RuntimeError("weights gone"))
        self.assertTrue(needs_memory("please remember my birthday"))
        self.assertFalse(needs_memory("how are you today"))


class LayaLifecycle(unittest.TestCase):
    def tearDown(self):
        LayaService.reset_instance()

    def test_singleton(self):
        a = LayaService.get_instance()
        b = LayaService.get_instance()
        self.assertIs(a, b)
        LayaService.reset_instance()
        self.assertIsNot(a, LayaService.get_instance())

    def test_predict_passthrough_then_unload(self):
        svc = LayaService(model_dir="fake-dir")
        svc._agent = FakeAgent(answers={"route": {"choice": "RAG"}})
        with patch.object(svc, "unload", wraps=svc.unload) as spy:
            res = svc.predict({"query": "q"}, {"route": ROUTE_QUESTION})
        self.assertEqual(res, {"answers": {"route": {"choice": "RAG"}}})
        spy.assert_called_once()
        self.assertFalse(svc.is_loaded())

    def test_predict_error_still_unloads(self):
        svc = LayaService(model_dir="fake-dir")
        svc._agent = FakeAgent(error=RuntimeError("boom"))
        with self.assertRaises(RuntimeError):
            svc.predict({"query": "q"}, {"route": ROUTE_QUESTION})
        self.assertFalse(svc.is_loaded())

    def test_load_failure_clear_message(self):
        svc = LayaService(model_dir="/nonexistent-laya-dir-xyz")
        with self.assertRaisesRegex(RuntimeError, "not found"):
            svc.load()
        with self.assertRaisesRegex(RuntimeError, "not found"):
            svc.predict({"query": "q"}, {"route": ROUTE_QUESTION})


class FrozenTemplates(unittest.TestCase):
    def test_route_matches_smoke_test(self):
        self.assertEqual(
            ROUTE_QUESTION["instructions"], "Which capability should serve this message?"
        )
        self.assertEqual(
            ROUTE_QUESTION["criteria"],
            {
                "DIRECT": "greetings, smalltalk, general knowledge answerable without files",
                "RAG": "needs attached local files or ingested documents",
                "WEB": "needs fresh internet info, current events, latest versions",
            },
        )

    def test_memory_matches_smoke_test(self):
        self.assertEqual(
            NEEDS_MEMORY_QUESTION["instructions"],
            "Does this need personal memory like name, preferences, or earlier conversation?",
        )

    def test_hit_grade_matches_smoke_test(self):
        self.assertEqual(
            HIT_GRADE_QUESTION["instructions"], "How relevant is this chunk to the query?"
        )
        self.assertEqual(HIT_GRADE_QUESTION["criteria"], ["irrelevant", "partial", "exact"])


def _grade_probs(level, conf=0.9):
    """Hit-grade probabilities with argmax at level, top prob conf."""
    rest = (1.0 - conf) / 2.0
    probs = {"0": rest, "1": rest, "2": rest}
    probs[str(level)] = conf
    return probs


class SeqAgent:
    """Per-hit canned hit_grade answers in call order, or a predict error."""

    def __init__(self, prob_list=None, error=None):
        self.probs = list(prob_list or [])
        self.error = error
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        if self.error is not None:
            raise self.error
        probs = self.probs[min(len(self.calls) - 1, len(self.probs) - 1)]
        return {"answers": {"hit_grade": {"probabilities": probs}}}


class LayaHitGrading(unittest.TestCase):
    def tearDown(self):
        LayaService.reset_instance()

    def _hit(self, idx, score, text=None):
        uid = uuid.uuid4()
        return SearchHit(
            id=f"{uid}:{idx}",
            document_id=str(uid),
            index=idx,
            text=text or f"chunk-{idx} body",
            score=score,
        )

    def search(self, hits, probs=None, error=None, threshold=0.8):
        engine = MagicMock()
        engine.embed.return_value = [[0.1] * 8]
        lance = MagicMock()
        fused = list(hits)
        lance.hybrid_search.return_value = fused
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [
            MagicMock(id=uuid.UUID(h.document_id), filename=f"{i}.md")
            for i, h in enumerate(fused)
        ]
        rag = rag_mod.RagService(db=db, engine=engine, lance=lance, docs=MagicMock())
        svc = LayaService(model_dir="fake-dir")
        fake = SeqAgent(prob_list=probs, error=error)
        svc._agent = fake
        inst = patch.object(rag_mod.LayaService, "get_instance", return_value=svc)
        inst.start()
        self.addCleanup(inst.stop)
        knob = patch.object(rag_mod.config, "LAYA_HIT_THRESHOLD", threshold)
        knob.start()
        self.addCleanup(knob.stop)
        self._svc = svc
        self._fake = fake
        self._fused = fused
        return rag

    def test_hit_threshold_knob_defaults(self):
        from config import Config

        self.assertEqual(Config.model_fields["LAYA_HIT_THRESHOLD"].default, 0.8)

    def test_exact_first_reorder_and_drop(self):
        a = self._hit(0, 0.9, text="aaa timetable")
        b = self._hit(1, 0.7, text="bbb timetable")
        c = self._hit(2, 0.5, text="ccc timetable")
        rag = self.search(
            [a, b, c],
            probs=[_grade_probs(1), _grade_probs(2), _grade_probs(0)],
        )
        with patch.object(self._svc, "load", wraps=self._svc.load) as load_spy:
            out = rag.search("nightly backup timetable")
        self.assertEqual([h.id for h in out], [b.id, a.id])
        load_spy.assert_called_once()  # one load serves the whole list
        self.assertFalse(self._svc.is_loaded())
        self.assertEqual(len(self._fake.calls), 3)
        state, questions = self._fake.calls[0]
        self.assertEqual(
            state,
            {
                "query": "nightly backup timetable",
                "chunk": "[0.md] aaa timetable",
                "filename": "0.md",
            },
        )
        self.assertEqual(set(questions), {"hit_grade"})
        self.assertIs(questions["hit_grade"], HIT_GRADE_QUESTION)

    def test_all_irrelevant_keeps_order(self):
        a = self._hit(0, 0.9)
        b = self._hit(1, 0.5)
        rag = self.search([a, b], probs=[_grade_probs(0), _grade_probs(0)])
        out = rag.search("q")
        self.assertEqual([h.id for h in out], [a.id, b.id])

    def test_below_threshold_advisory_keeps_order(self):
        a = self._hit(0, 0.9)
        b = self._hit(1, 0.5)
        rag = self.search(
            [a, b],
            probs=[{"0": 0.5, "1": 0.3, "2": 0.2}, {"0": 0.1, "1": 0.2, "2": 0.7}],
            threshold=0.8,
        )
        out = rag.search("q")
        self.assertEqual([h.id for h in out], [a.id, b.id])

    def test_laya_error_returns_fused_untouched(self):
        a = self._hit(0, 0.9)
        b = self._hit(1, 0.5)
        rag = self.search([a, b], error=RuntimeError("weights gone"))
        out = rag.search("q")
        self.assertIs(out, self._fused)

    def test_empty_hits_skip_predict(self):
        rag = self.search([], probs=[_grade_probs(2)])
        out = rag.search("q")
        self.assertEqual(out, [])
        self.assertEqual(self._fake.calls, [])

    def test_truncates_to_top_k(self):
        hits = [self._hit(i, 0.9 - i * 0.1) for i in range(3)]
        rag = self.search(hits, probs=[_grade_probs(2)] * 3)
        with patch.object(rag_mod, "top_k_for_ctx", return_value=2):
            out = rag.search("q")
        self.assertEqual([h.id for h in out], [hits[0].id, hits[1].id])

    def test_per_hit_fallback_without_predict_many(self):
        a = self._hit(0, 0.9)
        b = self._hit(1, 0.7)
        c = self._hit(2, 0.5)
        rag = self.search(
            [a, b, c],
            probs=[_grade_probs(1), _grade_probs(2), _grade_probs(0)],
        )
        self._svc.predict_many = None
        # Preset fakes are single-use under the real unload (it dels
        # _agent), so hold the fake resident to exercise the N-load loop.
        with patch.object(self._svc, "load", wraps=self._svc.load) as load_spy, patch.object(
            self._svc, "unload", lambda: None
        ):
            out = rag.search("q")
        self.assertEqual([h.id for h in out], [b.id, a.id])
        self.assertEqual(load_spy.call_count, 3)


class RewriteGate(unittest.TestCase):
    """L3 rewrite gate: Laya decides, SLM expands, retrieval uses final text.

    Laya is faked (no weights) and the SLM rewrite is faked; the graph
    runs with mocked rag/decider so only the gate wiring is exercised.
    """

    def tearDown(self):
        LayaService.reset_instance()

    def _run(self, query, history, noul=None, laya_error=None,
             rewrite_return=None, rewrite_error=None):
        from services import rag_graph as graph_mod
        from services.rag_graph import RagGraph

        seen = {}
        rag = MagicMock()
        rag.search.side_effect = lambda q, top_k=None, conversation_id=None: (
            seen.setdefault("search_q", q), []
        )[1]
        rag.build_messages.side_effect = lambda q, hits, hist, **kw: (
            seen.setdefault("build_q", q), [{"role": "user", "content": q}]
        )[1]
        rag.docs.list_by_conversation.return_value = [MagicMock(filename="n.md")]
        rag.docs.list_recent.return_value = [MagicMock(filename="n.md")]
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="RAG", reason="t")
        g = RagGraph(db=MagicMock(), rag=rag, decider=decider,
                     llm=MagicMock(), top_k=5)
        svc = LayaService(model_dir="fake-dir")
        answers = {"rewrite_needed": {"noul": noul}} if noul is not None else {}
        svc._agent = FakeAgent(answers=answers, error=laya_error)
        fake = svc._agent
        with patch.object(graph_mod.LayaService, "get_instance", return_value=svc), \
                patch.object(graph_mod, "_rewrite_via_slm") as rw, \
                patch.object(RagGraph, "_memory_text", return_value=None), \
                patch.object(RagGraph, "_topic_sibling_lines", return_value=[]):
            if rewrite_error is not None:
                rw.side_effect = rewrite_error
            else:
                rw.return_value = rewrite_return
            out = g.run(query, history, uuid.uuid4())
        return out, seen, fake, rw

    def _history(self):
        return [
            {"role": "user", "content": "What chunk size do we use?"},
            {"role": "assistant", "content": "800 tokens with 120 overlap."},
        ]

    def test_rewrite_threshold_knob_defaults(self):
        from config import Config

        self.assertEqual(Config.model_fields["LAYA_REWRITE_THRESHOLD"].default, 0.8)

    def test_rewrite_question_frozen(self):
        from services.rag_graph import REWRITE_QUESTION

        self.assertEqual(REWRITE_QUESTION["type"], "noul")
        self.assertEqual(
            REWRITE_QUESTION["instructions"],
            "Does this query need rewriting to be self-contained for retrieval?",
        )

    def test_gate_true_rewrites_search_keeps_build_original(self):
        out, seen, fake, rw = self._run(
            "and the overlap value?", self._history(), noul=0.9,
            rewrite_return="What is the chunk overlap value?",
        )
        self.assertEqual(len(fake.calls), 1)
        state, questions = fake.calls[0]
        self.assertEqual(set(questions), {"rewrite_needed"})
        self.assertEqual(state["query"], "and the overlap value?")
        self.assertIn("chunk size", state["history"])
        rw.assert_called_once()
        self.assertEqual(seen["search_q"], "What is the chunk overlap value?")
        self.assertEqual(seen["build_q"], "and the overlap value?")

    def test_gate_false_sends_original_zero_rewrite_cost(self):
        out, seen, fake, rw = self._run(
            "and the overlap value?", self._history(), noul=0.1,
            rewrite_return="SHOULD NOT BE USED",
        )
        rw.assert_not_called()
        self.assertEqual(seen["search_q"], "and the overlap value?")
        self.assertEqual(seen["build_q"], "and the overlap value?")

    def test_rewrite_slm_failure_falls_back_to_original(self):
        out, seen, fake, rw = self._run(
            "and the overlap value?", self._history(), noul=0.9,
            rewrite_error=RuntimeError("worker down"),
        )
        rw.assert_called_once()
        self.assertEqual(seen["search_q"], "and the overlap value?")
        self.assertEqual(seen["build_q"], "and the overlap value?")

    def test_empty_history_skips_gate(self):
        out, seen, fake, rw = self._run(
            "and the overlap value?", [], noul=0.9,
            rewrite_return="SHOULD NOT BE USED",
        )
        self.assertEqual(fake.calls, [])
        rw.assert_not_called()
        self.assertEqual(seen["search_q"], "and the overlap value?")

    def test_laya_error_skips_rewrite(self):
        out, seen, fake, rw = self._run(
            "and the overlap value?", self._history(),
            laya_error=RuntimeError("weights gone"),
            rewrite_return="SHOULD NOT BE USED",
        )
        rw.assert_not_called()
        self.assertEqual(seen["search_q"], "and the overlap value?")


class GroundingGate(unittest.TestCase):
    """L3 grounding gate: buffer RAG drafts, grade, stream or refuse."""

    def tearDown(self):
        LayaService.reset_instance()

    def _hit(self, uid, text="Lease: 2 cats allowed with $200 deposit.", heading="pets"):
        return SearchHit(
            id=f"{uid}:0",
            document_id=str(uid),
            index=0,
            heading=heading,
            text=text,
            score=0.9,
        )

    @staticmethod
    async def _drain(it):
        return [c if isinstance(c, str) else c.decode() for c in [c async for c in it]]

    @staticmethod
    def _events(chunks):
        out = []
        for part in "".join(chunks).split("\n\n"):
            for line in part.splitlines():
                if line.startswith("data: "):
                    out.append(line[len("data: "):])
        return out

    def _grade(self, laya_answers=None, laya_error=None, query="q", context="c", draft="d"):
        import services.grounding_service as ground_mod

        svc = LayaService(model_dir="fake-dir")
        fake = FakeAgent(answers=laya_answers, error=laya_error)
        svc._agent = fake
        with patch.object(ground_mod.LayaService, "get_instance", return_value=svc):
            out = ground_mod.grade_draft(query, context, draft)
        return out, fake

    def _run_chat(self, *, route, hits, deltas, laya_answers=None,
                  laya_error=None, filenames=None, watch_grade=False):
        import router.chat_api as capi
        import services.grounding_service as ground_mod
        from schemas.api_schemas import ChatRequest

        conv_id = uuid.uuid4()
        consumed = []
        saved = {}

        async def fake_stream():
            for d in deltas:
                consumed.append(d)
                yield d

        conv = MagicMock()
        conv.id = conv_id
        chat_svc = MagicMock()
        chat_svc.ensure_conversation.return_value = conv
        chat_svc.get_history.return_value = []
        chat_svc.run_agentic.return_value = {
            "messages": [{"role": "user", "content": "q"}],
            "route": route,
            "hits": hits,
        }

        def _add(cid, role, content):
            saved[role] = content
            return MagicMock()

        chat_svc.add_message.side_effect = _add
        fake_llm = MagicMock()
        fake_llm.astream_chat.side_effect = lambda **kw: fake_stream()
        fake_engine = MagicMock()
        fake_engine.is_generating.return_value = False

        db = MagicMock()
        rows = []
        for uid, name in (filenames or {}).items():
            row = MagicMock()
            row.id = uid
            row.filename = name
            rows.append(row)
        db.query.return_value.filter.return_value.all.return_value = rows

        svc = LayaService(model_dir="fake-dir")
        svc._agent = FakeAgent(answers=laya_answers, error=laya_error)

        grade_spy = None
        req = ChatRequest(query="What does the lease say about pets?", conversation_id=conv_id)

        async def _main():
            resp = await capi.chat(req, MagicMock(), db)
            at_response = list(consumed)
            chunks = await self._drain(resp.body_iterator)
            return resp, at_response, chunks

        with ExitStack() as stack:
            stack.enter_context(patch.object(capi, "ChatServices", return_value=chat_svc))
            stack.enter_context(patch.object(capi, "LLMService", return_value=fake_llm))
            stack.enter_context(
                patch.object(capi.LlamaEngine, "get_instance", return_value=fake_engine)
            )
            stack.enter_context(patch.object(capi.rollup_job, "submit", return_value=None))
            stack.enter_context(
                patch.object(ground_mod.LayaService, "get_instance", return_value=svc)
            )
            if watch_grade:
                grade_spy = MagicMock(wraps=ground_mod.grade_draft)
                stack.enter_context(patch.object(ground_mod, "grade_draft", grade_spy))
            resp, at_response, chunks = asyncio.run(_main())
        return {
            "response": resp,
            "chunks": chunks,
            "consumed": at_response,
            "saved": saved,
            "grade_spy": grade_spy,
        }

    def test_grounded_streams_buffered_in_order(self):
        uid = uuid.uuid4()
        out = self._run_chat(
            route="RAG",
            hits=[self._hit(uid)],
            deltas=["2 cats ", "allowed."],
            laya_answers={"is_grounded": {"noul": 0.95}},
            filenames={uid: "lease.txt"},
        )
        self.assertEqual(out["consumed"], ["2 cats ", "allowed."])  # full draft before first byte
        events = self._events(out["chunks"])
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(
            [json.loads(e)["content"] for e in events[:-1]], ["2 cats ", "allowed."]
        )
        self.assertEqual(out["saved"].get("assistant"), "2 cats allowed.")
        self.assertEqual(out["response"].headers.get("x-route"), "RAG")
        self.assertIn("x-ttft", {k.lower() for k in out["response"].headers.keys()})

    def test_ungrounded_substitutes_refusal(self):
        uid = uuid.uuid4()
        out = self._run_chat(
            route="RAG",
            hits=[self._hit(uid)],
            deltas=["3 dogs ", "allowed free."],
            laya_answers={"is_grounded": {"noul": 0.1}},
            filenames={uid: "lease.txt"},
        )
        events = self._events(out["chunks"])
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(len(events), 2)
        refusal = json.loads(events[0])["content"]
        self.assertIn("lease.txt", refusal)
        self.assertIn("rephrase", refusal)
        self.assertNotIn("3 dogs", "".join(out["chunks"]))
        self.assertEqual(out["saved"].get("assistant"), refusal)

    def test_laya_error_streams_original(self):
        uid = uuid.uuid4()
        out = self._run_chat(
            route="RAG",
            hits=[self._hit(uid)],
            deltas=["2 cats ", "allowed."],
            laya_error=RuntimeError("weights gone"),
            filenames={uid: "lease.txt"},
        )
        events = self._events(out["chunks"])
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(
            [json.loads(e)["content"] for e in events[:-1]], ["2 cats ", "allowed."]
        )
        self.assertEqual(out["saved"].get("assistant"), "2 cats allowed.")

    def test_direct_never_buffers(self):
        import services.grounding_service as ground_mod

        uid = uuid.uuid4()
        self.assertFalse(ground_mod.should_gate("DIRECT", [self._hit(uid)]))
        out = self._run_chat(
            route="DIRECT",
            hits=[self._hit(uid)],
            deltas=["hi ", "there ", "friend"],
            laya_answers={"is_grounded": {"noul": 0.1}},
            filenames={uid: "lease.txt"},
            watch_grade=True,
        )
        out["grade_spy"].assert_not_called()  # no grade call on DIRECT
        self.assertEqual(out["consumed"], ["hi "])  # first byte out before the rest exists
        events = self._events(out["chunks"])
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(
            [json.loads(e)["content"] for e in events[:-1]], ["hi ", "there ", "friend"]
        )
        self.assertEqual(out["saved"].get("assistant"), "hi there friend")

    def test_grade_true_false_none(self):
        out, fake = self._grade({"is_grounded": {"noul": 0.95}})
        self.assertTrue(out)
        out, _ = self._grade({"is_grounded": {"noul": 0.1}})
        self.assertFalse(out)
        out, _ = self._grade(laya_error=RuntimeError("weights gone"))
        self.assertIsNone(out)
        out, _ = self._grade({"is_grounded": {"noul": 0.95}}, draft="  ")
        self.assertIsNone(out)
        out, _ = self._grade({})
        self.assertIsNone(out)
        out, _ = self._grade({"is_grounded": {"noul": 0.8}})
        self.assertTrue(out)  # threshold is inclusive

    def test_grade_single_call_shape(self):
        import services.grounding_service as ground_mod

        out, fake = self._grade({"is_grounded": {"noul": 0.95}})
        self.assertTrue(out)
        self.assertEqual(len(fake.calls), 1)
        state, questions = fake.calls[0]
        self.assertEqual(set(state), {"query", "context", "answer"})
        self.assertEqual(set(questions), {"is_grounded"})
        self.assertIs(questions["is_grounded"], ground_mod.IS_GROUNDED_QUESTION)

    def test_ground_template_frozen(self):
        import services.grounding_service as ground_mod

        self.assertEqual(ground_mod.IS_GROUNDED_QUESTION["type"], "noul")
        self.assertEqual(
            ground_mod.IS_GROUNDED_QUESTION["instructions"],
            "Is this answer fully supported by the provided context, with no outside facts?",
        )

    def test_ground_threshold_knob_defaults(self):
        from config import Config

        self.assertEqual(Config.model_fields["LAYA_GROUND_THRESHOLD"].default, 0.8)

    def test_refusal_names_files(self):
        import services.grounding_service as ground_mod

        text = ground_mod.refusal_text(["b.md", "a.md"])
        self.assertIn("a.md, b.md", text)
        self.assertIn("rephrase", text)
        bare = ground_mod.refusal_text([])
        self.assertIn("cited files", bare)
        self.assertIn("rephrase", bare)

    def test_context_block_labels(self):
        import services.grounding_service as ground_mod

        uid = uuid.uuid4()
        did = str(uid)
        block = ground_mod.build_context_block(
            [self._hit(uid, text="body", heading="pets")], {did: "lease.txt"}
        )
        self.assertIn("[lease.txt:pets]\nbody", block)
        page_hit = SearchHit(id="x", document_id=did, text="p", page=3, score=0.1)
        self.assertIn("[lease.txt:p3]", ground_mod.build_context_block([page_hit], {did: "lease.txt"}))
        plain_hit = SearchHit(id="x", document_id=did, text="p", score=0.1)
        self.assertIn("[lease.txt]\n", ground_mod.build_context_block([plain_hit], {did: "lease.txt"}))
        self.assertIn(f"[{did}]", ground_mod.build_context_block([plain_hit]))

    def test_should_gate(self):
        import services.grounding_service as ground_mod

        hit = self._hit(uuid.uuid4())
        self.assertTrue(ground_mod.should_gate("RAG", [hit]))
        self.assertFalse(ground_mod.should_gate("RAG", []))
        self.assertFalse(ground_mod.should_gate("DIRECT", [hit]))
        self.assertFalse(ground_mod.should_gate("DIRECT", []))


if __name__ == "__main__":
    unittest.main()
