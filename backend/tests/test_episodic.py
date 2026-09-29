"""Episodic memory tests: rollup write, recall read, budget sub-caps.

Runs offline (no GGUF): memory SQLite, mocked summarize + engines.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, ConversationsModel, EpisodicMemoryModel
from repository.episodic_repository import EpisodicMemoryRepository
from repository.semantic_repository import SemanticMemoryRepository
from schemas.rag_schemas import RouteDecision
from services.episodic_service import EpisodicService


class EpisodicCase(unittest.TestCase):
    def setUp(self):
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()

    def tearDown(self):
        self.db.close()

    def conv(self):
        row = ConversationsModel(title="t")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row.id

    def turns(self, n, tag="topic"):
        roles = ["user", "assistant"]
        return [
            {"role": roles[i % 2], "content": f"{tag}-{i} short note"}
            for i in range(n)
        ]


class EpisodicRepo(EpisodicCase):
    def test_create_and_ordering(self):
        repo = EpisodicMemoryRepository(self.db)
        cid = self.conv()
        repo.create(cid, "first summary", 0, 8)
        repo.create(cid, "second summary", 8, 16)
        chrono = repo.list_by_conversation(cid, limit=10)
        self.assertEqual([r.summary for r in chrono], ["first summary", "second summary"])
        self.assertEqual((chrono[0].turn_start, chrono[0].turn_end), (0, 8))
        recent = repo.list_recent_for_query(cid, limit=10)
        self.assertEqual([r.summary for r in recent], ["second summary", "first summary"])

    def test_delete_by_conversation(self):
        repo = EpisodicMemoryRepository(self.db)
        cid, other = self.conv(), self.conv()
        repo.create(cid, "gone", 0, 8)
        repo.create(other, "kept", 0, 8)
        self.assertEqual(repo.delete_by_conversation(cid), 1)
        self.assertEqual(repo.list_by_conversation(cid), [])
        self.assertEqual(len(repo.list_by_conversation(other)), 1)


class EpisodicRecall(EpisodicCase):
    def test_overlap_ranking_recency_tiebreak(self):
        cid = self.conv()
        EpisodicService(self.db).repo.create(cid, "project deadline is friday", 0, 4)
        EpisodicService(self.db).repo.create(cid, "user likes ramen noodles", 4, 8)
        out = EpisodicService(self.db).recall(cid, "what ramen to cook?", limit=3)
        self.assertTrue(out[0].startswith("user likes ramen"))
        self.assertEqual(len(out), 2)

    def test_recall_limit_and_strings(self):
        cid = self.conv()
        repo = EpisodicMemoryRepository(self.db)
        for i in range(4):
            repo.create(cid, f"note about topic {i}", i, i + 1)
        out = EpisodicService(self.db).recall(cid, "zzz no overlap qqq", limit=2)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(s, str) and s.strip() for s in out))

    def test_recall_failure_fail_open(self):
        svc = EpisodicService(self.db)
        with patch.object(
            EpisodicMemoryRepository, "list_recent_for_query", side_effect=RuntimeError("db down")
        ):
            self.assertEqual(svc.recall(self.conv(), "ramen", limit=3), [])


class EpisodicRollup(EpisodicCase):
    def test_no_trigger_short_history(self):
        import services.summarize_service as sum_mod

        svc = EpisodicService(self.db)
        with patch.object(sum_mod, "summarize_text") as m:
            self.assertIsNone(svc.rollup_if_needed(self.conv(), self.turns(5), 100000))
        m.assert_not_called()
        self.assertEqual(self.db.query(EpisodicMemoryModel).count(), 0)

    def test_count_trigger_skips_last_two(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        svc = EpisodicService(self.db)
        with patch.object(sum_mod, "summarize_text", return_value="rolled") as m:
            out = svc.rollup_if_needed(cid, self.turns(10), 100000)
        self.assertEqual(out, "rolled")
        head_text = m.call_args[0][0]
        self.assertIn("topic-0", head_text)
        self.assertNotIn("topic-8", head_text)
        self.assertNotIn("topic-9", head_text)
        self.assertEqual(m.call_args[1].get("max_tokens"), 128)
        rows = EpisodicMemoryRepository(self.db).list_by_conversation(cid)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].turn_start, rows[0].turn_end), (0, 8))

    def test_overflow_trigger(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        big = [{"role": "user", "content": "word " * 2000} for _ in range(5)]
        svc = EpisodicService(self.db)
        with patch.object(sum_mod, "summarize_text", return_value="overflow roll"):
            out = svc.rollup_if_needed(cid, big, 100)
        self.assertEqual(out, "overflow roll")
        self.assertEqual(len(EpisodicMemoryRepository(self.db).list_by_conversation(cid)), 1)

    def test_busy_returns_none(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        svc = EpisodicService(self.db)
        busy = MagicMock()
        busy.is_generating.return_value = True
        with (
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=busy),
            patch.object(sum_mod, "summarize_text") as m,
        ):
            self.assertIsNone(svc.rollup_if_needed(cid, self.turns(10), 100000))
        m.assert_not_called()
        self.assertEqual(EpisodicMemoryRepository(self.db).list_by_conversation(cid), [])

    def test_summarize_failure_returns_none(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        svc = EpisodicService(self.db)
        idle = MagicMock()
        idle.is_generating.return_value = False
        with (
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", side_effect=RuntimeError("Busy")),
        ):
            self.assertIsNone(svc.rollup_if_needed(cid, self.turns(10), 100000))
        self.assertEqual(EpisodicMemoryRepository(self.db).list_by_conversation(cid), [])


class EpisodicGraph(EpisodicCase):
    def graph(self, rag):
        from services.rag_graph import RagGraph

        llm = MagicMock()
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        return RagGraph(db=self.db, rag=rag, decider=decider, llm=llm)

    def test_episodic_failure_semantic_only(self):
        SemanticMemoryRepository(self.db).upsert("name", "Ada")
        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(cid, "user likes ramen", 0, 8)
        rag = MagicMock()
        rag.db = self.db
        rag.build_messages.side_effect = lambda q, h, hist, **kw: [{"role": "u", "content": kw.get("memory_text") or ""}]
        import services.episodic_service as epi_mod

        with patch.object(epi_mod.EpisodicService, "recall", side_effect=RuntimeError("down")):
            out = self.graph(rag).run("what is my name?", [{"role": "user", "content": "hi"}], cid)
        mem = rag.build_messages.call_args[1].get("memory_text")
        self.assertIn("Ada", mem)
        self.assertNotIn("earlier:", mem)
        self.assertEqual(out["route"], "DIRECT")

    def test_mem_sub_cap_respected(self):
        from config import config
        from core.context_budget import count_tokens

        SemanticMemoryRepository(self.db).upsert("name", "Ada " * 5000)
        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(cid, "user likes ramen noodles a lot", 0, 8)
        rag = MagicMock()
        rag.db = self.db
        rag.build_messages.return_value = [{"role": "u", "content": "x"}]
        from services.rag_graph import RagGraph

        llm = MagicMock()
        decider = MagicMock()
        decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
        text = RagGraph(db=self.db, rag=rag, decider=decider, llm=llm)._memory_text(
            "what is my name?", cid
        )
        self.assertIn("name:", text)
        self.assertLessEqual(count_tokens(text), config.EFFECTIVE_MEMORY_TOKENS)

    def test_combined_semantic_first(self):
        from services.rag_graph import _combine_memory

        big_sem = "name: " + "Ada " * 5000
        out = _combine_memory(big_sem, ["user likes ramen"])
        self.assertIn("name:", out)
        self.assertNotIn("ramen", out)


class EpisodicCascade(EpisodicCase):
    def test_delete_conversation_removes_episodic(self):
        from services.chat_services import ChatServices
        import services.chat_services as chat_mod

        cid = self.conv()
        EpisodicMemoryRepository(self.db).create(cid, "old summary", 0, 8)
        fake_rag = MagicMock()
        fake_rag.docs.list_by_conversation.return_value = []
        with patch.object(chat_mod, "RagService", return_value=fake_rag):
            ChatServices(self.db).delete_conversation(cid)
        self.assertEqual(EpisodicMemoryRepository(self.db).list_by_conversation(cid), [])
        self.assertIsNone(self.db.query(ConversationsModel).filter_by(id=cid).first())

    def test_maybe_rollup_fail_open(self):
        from services.chat_services import ChatServices

        cid = self.conv()
        self.assertIsNone(ChatServices(self.db).maybe_rollup(uuid.uuid4()))
        self.assertIsNone(ChatServices(self.db).maybe_rollup(cid))


if __name__ == "__main__":
    unittest.main()
