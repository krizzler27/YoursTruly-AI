"""Laya L1 tests: gating policy with a faked agent, no weights.

Covers the decider swap (route plus needs-memory in one call, threshold
gating, safe-defaults, SLM fallback) and the LayaService lifecycle
(singleton, transient load/unload, clear load errors). Never touches the
real backend/laya-model weights.
"""

import sys
import unittest
import uuid
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


if __name__ == "__main__":
    unittest.main()
