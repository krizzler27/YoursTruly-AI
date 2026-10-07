"""Model-free worker slot tests: roles, resolve, busy skip, timeout, unload.

Runs offline (no GGUF): fake engines stand in for llama.cpp weights.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import time
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from unittest.mock import PropertyMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config as config_obj
import services.llama_engine as engine_mod
import services.summarize_service as summ_mod
from services.llama_engine import (
    EmbeddingEngine,
    LlamaEngine,
    WorkerEngine,
    resolve_worker_model_path,
)


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


class RoleSlots(TestCase):
    def test_singleton_per_role_and_lazy(self):
        chat = LlamaEngine.get_instance("chat")
        self.assertIs(chat, LlamaEngine.get_instance("chat"))
        worker = LlamaEngine.get_instance("worker")
        self.assertIsInstance(worker, WorkerEngine)
        self.assertIs(worker, WorkerEngine.get_instance("worker"))
        self.assertIsNot(chat, worker)
        embed = EmbeddingEngine.get_instance("embed")
        self.assertIsNot(embed, worker)
        self.assertFalse(worker.is_loaded())  # slot created empty, never loads


class ResolveWorker(TestCase):
    def test_fallback_without_weights(self):
        with patch.object(engine_mod, "list_models", return_value=[]), \
            patch.object(config_obj, "LLAMA_WORKER_MODEL", None), \
            patch.object(
                type(config_obj), "EFFECTIVE_CHAT_MODEL",
                new_callable=PropertyMock, return_value="/tmp/fake-chat.gguf",
            ):
            path, fallback = resolve_worker_model_path()
        self.assertEqual(path, "/tmp/fake-chat.gguf")
        self.assertTrue(fallback)

    def test_qwen_pick_before_chat_fallback(self):
        qwen = Path("/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
        with patch.object(engine_mod, "list_models", return_value=[qwen]), \
            patch.object(config_obj, "LLAMA_WORKER_MODEL", None):
            path, fallback = resolve_worker_model_path()
        self.assertEqual(path, str(qwen))
        self.assertFalse(fallback)


class SummarizeChatPath(TestCase):
    def test_busy_check_skips_without_invoke(self):
        llm_cls = patch.object(summ_mod, "LLMService")
        eng_cls = patch.object(summ_mod, "LlamaEngine")
        with llm_cls as mock_llm, eng_cls as mock_eng:
            mock_eng.get_instance.return_value = FakeEngine(generating=True)
            out = summ_mod.summarize_text("line one\nline two", max_tokens=10, role="chat")
            mock_llm.assert_not_called()
        self.assertEqual(out, "line one / line two")

    def test_chat_success_keeps_resident_slot(self):
        fake = FakeEngine(generating=False)
        llm = patch.object(summ_mod, "LLMService")
        eng = patch.object(summ_mod, "LlamaEngine")
        with llm as mock_llm, eng as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "  hello summary  "
            out = summ_mod.summarize_text("some long text here", max_tokens=32, role="chat")
        self.assertEqual(out, "hello summary")
        self.assertEqual(fake.unloads, 0)  # chat stays resident, never unloaded

    def test_timeout_falls_back_to_extractive(self):
        fake = FakeEngine(generating=False)

        def slow(*args, **kwargs):
            time.sleep(0.3)
            return "too late"

        llm = patch.object(summ_mod, "LLMService")
        eng = patch.object(summ_mod, "LlamaEngine")
        with llm as mock_llm, eng as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = slow
            out = summ_mod.summarize_text(
                "line one\nline two", max_tokens=10, timeout=0.05, role="chat"
            )
        self.assertEqual(out, "line one / line two")


class SummarizeWorkerPath(TestCase):
    def test_worker_unloads_after_success(self):
        fake = FakeEngine(generating=False)
        llm = patch.object(summ_mod, "LLMService")
        worker = patch.object(summ_mod, "WorkerEngine")
        with llm as mock_llm, worker as mock_worker:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "worker summary"
            out = summ_mod.summarize_text("some text", max_tokens=32, role="worker")
        self.assertEqual(out, "worker summary")
        self.assertTrue(fake.loaded)
        self.assertEqual(fake.unloads, 1)

    def test_worker_failure_falls_back_to_extractive(self):
        fake = FakeEngine(generating=False)
        llm = patch.object(summ_mod, "LLMService")
        worker = patch.object(summ_mod, "WorkerEngine")
        with llm as mock_llm, worker as mock_worker:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.side_effect = RuntimeError("boom")
            out = summ_mod.summarize_text("line one\nline two", max_tokens=10, role="worker")
        self.assertEqual(out, "line one / line two")
        self.assertEqual(fake.unloads, 1)  # unload still runs after failure


class SummarizeAutoRoute(TestCase):
    """resolve_slot auto-route: RAM x worker-file x explicit-role matrix."""

    def test_auto_picks_worker_when_ram_and_file(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=14.0), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/qwen2.5-1.5b.gguf", False)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot(), "worker")
            fake = FakeEngine(generating=False)
            with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
                patch.object(summ_mod, "LLMService") as mock_llm:
                mock_worker.get_instance.return_value = fake
                mock_llm.return_value.invoke.return_value = "auto worker"
                out = summ_mod.summarize_text("some text here", max_tokens=32)
            self.assertEqual(out, "auto worker")
            self.assertEqual(fake.unloads, 1)

    def test_auto_picks_worker_at_13_9(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=13.9), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/qwen2.5-1.5b.gguf", False)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot(), "worker")

    def test_auto_falls_back_without_worker_file(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=14.0), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/chat.gguf", True)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot(), "chat")
            fake = FakeEngine(generating=False)
            with patch.object(summ_mod, "LlamaEngine") as mock_eng, \
                patch.object(summ_mod, "LLMService") as mock_llm:
                mock_eng.get_instance.return_value = fake
                mock_llm.return_value.invoke.return_value = "chat summary"
                out = summ_mod.summarize_text("some text here", max_tokens=32)
            self.assertEqual(out, "chat summary")

    def test_auto_falls_back_on_low_ram(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=7.9), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/qwen2.5-1.5b.gguf", False)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot(), "chat")

    def test_auto_falls_back_below_threshold(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=11.9), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/qwen2.5-1.5b.gguf", False)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot(), "chat")

    def test_explicit_chat_wins_on_16gb(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=32.0), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/qwen2.5-1.5b.gguf", False)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot("chat"), "chat")
            fake = FakeEngine(generating=False)
            with patch.object(summ_mod, "LlamaEngine") as mock_eng, \
                patch.object(summ_mod, "LLMService") as mock_llm, \
                patch.object(summ_mod, "WorkerEngine") as mock_worker:
                mock_eng.get_instance.return_value = fake
                mock_llm.return_value.invoke.return_value = "chat wins"
                out = summ_mod.summarize_text("some text here", max_tokens=32, role="chat")
            self.assertEqual(out, "chat wins")
            mock_worker.get_instance.assert_not_called()

    def test_explicit_worker_honored(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=8.0), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/chat.gguf", True)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "chat"):
            self.assertEqual(summ_mod.resolve_slot("worker"), "worker")
            fake = FakeEngine(generating=False)
            with patch.object(summ_mod, "WorkerEngine") as mock_worker, \
                patch.object(summ_mod, "LLMService") as mock_llm:
                mock_worker.get_instance.return_value = fake
                mock_llm.return_value.invoke.return_value = "forced worker"
                out = summ_mod.summarize_text("some text here", max_tokens=32, role="worker")
            self.assertEqual(out, "forced worker")

    def test_configured_non_default_respected(self):
        with patch.object(summ_mod, "total_ram_gb", return_value=8.0), \
            patch.object(summ_mod, "resolve_worker_model_path",
                         return_value=("/models/chat.gguf", True)), \
            patch.object(config_obj, "SUMMARY_MODEL_ROLE", "worker"):
            self.assertEqual(summ_mod.resolve_slot(), "worker")


class SummarizeDegenerate(TestCase):
    def test_is_degenerate_repetition(self):
        self.assertTrue(summ_mod.is_degenerate("I I I I I ..."))
        self.assertTrue(summ_mod.is_degenerate("go go go go stop"))
        self.assertTrue(summ_mod.is_degenerate("Go GO go GO onward"))
        self.assertFalse(summ_mod.is_degenerate("go go go stop now"))
        self.assertFalse(summ_mod.is_degenerate(""))
        self.assertFalse(summ_mod.is_degenerate("   "))
        self.assertFalse(summ_mod.is_degenerate("hello summary"))

    def test_is_degenerate_low_diversity(self):
        low = ("alpha beta " * 25).strip()
        self.assertTrue(summ_mod.is_degenerate(low))
        varied = " ".join(f"word-{i}" for i in range(50))
        self.assertFalse(summ_mod.is_degenerate(varied))

    def test_chat_degenerate_falls_back_to_extractive(self):
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "LlamaEngine") as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "I I I I I I ..."
            out = summ_mod.summarize_text("line one\nline two", max_tokens=10, role="chat")
        self.assertEqual(out, "line one / line two")

    def test_worker_degenerate_falls_back_to_extractive(self):
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "WorkerEngine") as mock_worker:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "I I I I I I ..."
            out = summ_mod.summarize_text("line one\nline two", max_tokens=10, role="worker")
        self.assertEqual(out, "line one / line two")
        self.assertEqual(fake.unloads, 1)

    def test_worker_low_diversity_falls_back(self):
        fake = FakeEngine(generating=False)
        low = ("alpha beta " * 25).strip()
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "WorkerEngine") as mock_worker:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = low
            out = summ_mod.summarize_text("line one\nline two", max_tokens=60, role="worker")
        self.assertEqual(out, "line one / line two")

    def test_near_threshold_normal_passes_through(self):
        normal = "go go go then we left for the market with fresh bread and cheese"
        self.assertFalse(summ_mod.is_degenerate(normal))
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "LlamaEngine") as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = normal
            out = summ_mod.summarize_text("some long text here", max_tokens=32, role="chat")
        self.assertEqual(out, normal)

    def test_chat_invoke_kwargs_temp_and_penalty(self):
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "LlamaEngine") as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "a fine normal summary here"
            summ_mod.summarize_text("some long text here", max_tokens=32, role="chat")
        kwargs = mock_llm.return_value.invoke.call_args[1]
        self.assertEqual(kwargs.get("temperature"), 0.6)
        self.assertEqual(kwargs.get("repeat_penalty"), 1.1)

    def test_worker_invoke_kwargs_temp_and_penalty(self):
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "WorkerEngine") as mock_worker:
            mock_worker.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "a fine normal summary here"
            summ_mod.summarize_text("some long text here", max_tokens=32, role="worker")
        kwargs = mock_llm.return_value.invoke.call_args[1]
        self.assertEqual(kwargs.get("temperature"), 0.6)
        self.assertEqual(kwargs.get("repeat_penalty"), 1.1)

    def test_timeout_path_output_screened(self):
        fake = FakeEngine(generating=False)
        with patch.object(summ_mod, "LLMService") as mock_llm, \
            patch.object(summ_mod, "LlamaEngine") as mock_eng:
            mock_eng.get_instance.return_value = fake
            mock_llm.return_value.invoke.return_value = "I I I I I I ..."
            out = summ_mod.summarize_text(
                "line one\nline two", max_tokens=10, timeout=5.0, role="chat"
            )
        self.assertEqual(out, "line one / line two")

    def test_blank_inputs_return_empty(self):
        self.assertEqual(summ_mod.summarize_text("", role="chat"), "")
        self.assertEqual(summ_mod.summarize_text("   \n  ", role="worker"), "")
