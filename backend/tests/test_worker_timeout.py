"""Worker-slot timeout: hung SLM abandons with fail-open skip.

Runs offline (no GGUF): fake engines stand in for llama.cpp weights.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base
import services.summarize_service as summ_mod
from services import semantic_writer


class FakeEngine:
    """Stand-in slot: records load/unload, never touches weights."""

    def __init__(self, generating=False):
        self._generating = generating
        self.loaded = False
        self.unloads = 0

    def is_generating(self):
        return self._generating

    def ensure_loaded(self):
        self.loaded = True

    def unload(self):
        self.unloads += 1


class SummarizeWorkerTimeout(TestCase):
    def test_hang_skips_fail_open_and_next_usable(self):
        """Hung worker SLM abandons after budget, next job still runs."""
        fake = FakeEngine()

        def slow(*args, **kwargs):
            time.sleep(0.5)
            return "slow summary should be abandoned"

        with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
            patch.object(summ_mod, "LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = slow
            start = time.monotonic()
            with self.assertLogs("services.summarize_service", level="WARNING") as logs:
                out = summ_mod.summarize_text(
                    "line one\nline two", max_tokens=10, timeout=0.05, role="worker"
                )
            elapsed = time.monotonic() - start
        self.assertEqual(out, "line one / line two")
        self.assertLess(elapsed, 0.4)
        self.assertIn("summary_timeout", "\n".join(logs.output))
        time.sleep(0.6)
        fast = FakeEngine()
        with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
            patch.object(summ_mod, "LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fast
            mock_llm.return_value.invoke.return_value = "next ok here"
            out2 = summ_mod.summarize_text(
                "another text here", max_tokens=32, timeout=2.0, role="worker"
            )
        self.assertEqual(out2, "next ok here")

    def test_busy_skips_fail_open(self):
        """Busy worker keeps current skip behavior."""
        fake = FakeEngine()
        with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
            patch.object(summ_mod, "LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = RuntimeError(
                "System Busy - model is generating. Try again."
            )
            out = summ_mod.summarize_text(
                "line one\nline two", max_tokens=10, timeout=2.0, role="worker"
            )
        self.assertEqual(out, "line one / line two")

    def test_empty_falls_back(self):
        """Empty worker output keeps current fallback behavior."""
        fake = FakeEngine()
        with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
            patch.object(summ_mod, "LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = ""
            out = summ_mod.summarize_text(
                "line one\nline two", max_tokens=10, timeout=2.0, role="worker"
            )
        self.assertEqual(out, "line one / line two")


class ExtractWorkerTimeout(TestCase):
    def test_hang_skips_fail_open(self):
        """Hung extraction abandons after budget and returns []."""
        from schemas.rag_schemas import SemanticFactList

        fake = FakeEngine()

        def slow(*args, **kwargs):
            time.sleep(0.5)
            return SemanticFactList.model_validate({"facts": [{"key": "k", "value": "v"}]})

        with patch("services.llama_engine.WorkerEngine") as mock_worker, \
            patch("services.llm_service.LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = slow
            start = time.monotonic()
            with self.assertLogs("services.semantic_writer", level="WARNING") as logs:
                out = semantic_writer.extract_facts_slm(["I like coffee"], timeout=0.05)
            elapsed = time.monotonic() - start
        self.assertEqual(out, [])
        self.assertLess(elapsed, 0.4)
        self.assertTrue(any("timeout" in m.lower() for m in logs.output))
        time.sleep(0.6)

    def test_busy_and_empty_skip(self):
        """Busy and empty extraction keep current skip behavior."""
        fake = FakeEngine()
        with patch("services.llama_engine.WorkerEngine") as mock_worker, \
            patch("services.llm_service.LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = RuntimeError(
                "System Busy - model is generating. Try again."
            )
            self.assertEqual(semantic_writer.extract_facts_slm(["I like coffee"]), [])
        self.assertEqual(semantic_writer.extract_facts_slm([]), [])
        self.assertEqual(semantic_writer.extract_facts_slm(["   "]), [])


class AggregateWorkerTimeout(TestCase):
    def setUp(self):
        """Temp DB per test, never the real yourstrulyai.db."""
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()

    def tearDown(self):
        self.db.close()

    def test_empty_skips_with_none(self):
        """Empty aggregate keeps current skip behavior."""
        from services import project_service

        from services.chat_services import ChatServices

        cid = ChatServices(self.db).ensure_conversation(None, title="solo").id
        self.assertIsNone(project_service.aggregate_project(self.db, cid))

    def test_hang_bounded_fail_open(self):
        """Hung aggregate abandons after budget and stays usable."""
        from services import project_service
        from services.chat_services import ChatServices
        from repository.episodic_repository import EpisodicMemoryRepository

        cid = ChatServices(self.db).ensure_conversation(None, title="one").id
        from repository.project_repository import ProjectRepository

        proj = ProjectRepository(self.db).get_or_create("laya")
        ChatServices(self.db).set_project(cid, proj.id)
        EpisodicMemoryRepository(self.db).create(cid, "decided launch friday", 0, 4)

        def slow(*args, **kwargs):
            time.sleep(0.5)
            return "slow aggregate should be abandoned"

        fake = FakeEngine()
        with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
            patch.object(summ_mod, "LLMService") as mock_llm:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = slow
            start = time.monotonic()
            out = project_service.aggregate_project(self.db, cid, timeout=0.05)
            elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.4)
        self.assertIsNotNone(out)
        time.sleep(0.6)
