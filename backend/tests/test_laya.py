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

from schemas.rag_schemas import RouteDecision
from services import decider as dec_mod
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


if __name__ == "__main__":
    unittest.main()
