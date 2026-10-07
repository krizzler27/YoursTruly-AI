"""Episodic memory tests: rollup write, recall read, budget sub-caps.

Runs offline (no GGUF): memory SQLite, mocked summarize + engines.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, ConversationsModel, EpisodicMemoryModel, MessagesModel
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

    def test_recall_vector_rescues_paraphrase(self):
        cid = self.conv()
        coffee = EpisodicService(self.db).repo.create(
            cid, "user prefers strong coffee", 0, 4
        )
        friday = EpisodicService(self.db).repo.create(
            cid, "project deadline is friday", 4, 8
        )
        q = "morning brew habit"
        plain = EpisodicService(self.db).recall(cid, q, limit=3)
        self.assertTrue(plain[0].startswith("project deadline"))
        out = EpisodicService(self.db).recall(
            cid, q, limit=3,
            query_vector=[1.0, 0.0],
            candidate_vectors={str(coffee.id): [1.0, 0.0], str(friday.id): [0.0, 1.0]},
        )
        self.assertEqual(out, ["user prefers strong coffee"])

    def test_recall_overlap_stays_first_with_vector(self):
        cid = self.conv()
        coffee = EpisodicService(self.db).repo.create(
            cid, "user prefers strong coffee", 0, 4
        )
        friday = EpisodicService(self.db).repo.create(
            cid, "project deadline is friday", 4, 8
        )
        out = EpisodicService(self.db).recall(
            cid, "what coffee to brew?", limit=3,
            query_vector=[0.0, 1.0],
            candidate_vectors={str(coffee.id): [1.0, 0.0], str(friday.id): [0.0, 1.0]},
        )
        self.assertEqual(
            out, ["user prefers strong coffee", "project deadline is friday"]
        )

    def test_recall_vector_below_threshold_returns_none(self):
        cid = self.conv()
        EpisodicService(self.db).repo.create(cid, "user prefers strong coffee", 0, 4)
        EpisodicService(self.db).repo.create(cid, "project deadline is friday", 4, 8)
        rows = EpisodicService(self.db).repo.list_recent_for_query(cid, limit=10)
        cands = {str(r.id): [0.0, 1.0] for r in rows}
        out = EpisodicService(self.db).recall(
            cid, "morning brew habit", limit=3,
            query_vector=[1.0, 0.0], candidate_vectors=cands,
        )
        self.assertEqual(out, [])


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


class RollupJobQueue(EpisodicCase):
    def seed_history(self, cid, n=10):
        for i in range(n):
            self.db.add(
                MessagesModel(
                    conversation_id=cid,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"topic-{i} short note",
                )
            )
        self.db.commit()

    def test_submit_enqueues_fast_without_worker(self):
        import services.rollup_job as rollup_mod

        cid = self.conv()
        t0 = time.perf_counter()
        with patch.object(rollup_mod._runner, "submit") as queued:
            rollup_mod.rollup_job.submit(self.db, cid)
        elapsed = time.perf_counter() - t0
        queued.assert_called_once()
        payload = queued.call_args[0][0]
        self.assertEqual(payload[0], str(cid))
        self.assertEqual(self.db.query(EpisodicMemoryModel).count(), 0)
        self.assertLess(elapsed, 1.0)

    def test_submit_unknown_conversation_skips(self):
        import services.rollup_job as rollup_mod

        with patch.object(rollup_mod._runner, "submit") as queued:
            rollup_mod.rollup_job.submit(self.db, uuid.uuid4())
        queued.assert_not_called()
        self.assertEqual(self.db.query(EpisodicMemoryModel).count(), 0)

    def test_run_persists_rollup(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod

        cid = self.conv()
        self.seed_history(cid, 10)
        idle = MagicMock()
        idle.is_generating.return_value = False
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", return_value="rolled"),
        ):
            rollup_mod.rollup_job.run((str(cid), "test-req"))
        rows = EpisodicMemoryRepository(self.db).list_by_conversation(cid)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].summary, "rolled")
        self.assertEqual((rows[0].turn_start, rows[0].turn_end), (0, 8))

    def test_run_short_history_stores_nothing(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod

        cid = self.conv()
        self.seed_history(cid, 2)
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch.object(sum_mod, "summarize_text", return_value="rolled") as m,
        ):
            rollup_mod.rollup_job.run((str(cid), "test-req"))
        m.assert_not_called()
        self.assertEqual(EpisodicMemoryRepository(self.db).list_by_conversation(cid), [])


class RollupSseOrdering(EpisodicCase):
    @staticmethod
    async def _drain(it):
        return [c async for c in it]

    def test_save_then_submit_then_done(self):
        import asyncio
        import router.chat_api as chat_api_mod
        from schemas.api_schemas import ChatRequest

        cid = self.conv()
        events = []

        class FakeChat:
            def __init__(self, db):
                pass

            def ensure_conversation(self, conversation_id, title=None):
                conv = MagicMock()
                conv.id = conversation_id or cid
                return conv

            def get_history(self, conversation_id, limit=5):
                return []

            def run_agentic(self, query, history, conversation_id):
                return {"messages": [{"role": "user", "content": query}], "route": "DIRECT"}

            def add_message(self, conversation_id, role, content):
                events.append(("save", role))
                return MagicMock()

        async def fake_stream():
            yield "hello"

        fake_llm = MagicMock()
        fake_llm.astream_chat.return_value = fake_stream()
        fake_engine = MagicMock()
        fake_engine.is_generating.return_value = False

        def fake_submit(db, conversation_id):
            events.append(("rollup", str(conversation_id)))

        req = ChatRequest(query="hi", conversation_id=cid)
        with (
            patch.object(chat_api_mod, "ChatServices", FakeChat),
            patch.object(chat_api_mod, "LLMService", return_value=fake_llm),
            patch.object(chat_api_mod.LlamaEngine, "get_instance", return_value=fake_engine),
            patch.object(chat_api_mod.rollup_job, "submit", side_effect=fake_submit) as submitted,
        ):
            resp = asyncio.run(chat_api_mod.chat(req, MagicMock(), self.db))
            chunks = asyncio.run(self._drain(resp.body_iterator))
        submitted.assert_called_once()
        self.assertEqual(
            events, [("save", "user"), ("save", "assistant"), ("rollup", str(cid))]
        )
        self.assertTrue(chunks)
        self.assertIn("hello", chunks[0])
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")


class EpisodicCompaction(EpisodicCase):
    def seed_rows(self, cid, n):
        repo = EpisodicMemoryRepository(self.db)
        for i in range(n):
            repo.create(
                cid,
                f"summary number {i} with extra words to grow tokens",
                i * 8,
                (i + 1) * 8,
            )
        return repo.list_ordered_for_compaction(cid, limit=1000)

    def test_merge_oldest_pair_contiguous_span(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        self.seed_rows(cid, 21)
        svc = EpisodicService(self.db)
        with patch.object(sum_mod, "summarize_text", return_value="merged ab"):
            stats = svc.compact_if_needed(cid)
        self.assertEqual(stats["compacted"], 1)
        self.assertEqual(stats["merged_turns"], "0-16")
        rows = EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
            cid, limit=1000
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual((rows[0].turn_start, rows[0].turn_end), (0, 16))
        self.assertEqual(rows[0].summary, "merged ab")
        spans = [(r.turn_start, r.turn_end) for r in rows]
        self.assertEqual(spans, sorted(spans))

    def test_merge_abort_empty_keeps_originals(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        rows = self.seed_rows(cid, 21)
        # Make forget ineligible so the abort assertion isolates the merge.
        for r in rows[:-10]:
            r.recall_count = 1
        self.db.commit()
        oldest = [(r.turn_start, r.turn_end) for r in rows[:2]]
        svc = EpisodicService(self.db)
        with patch.object(sum_mod, "summarize_text", return_value=""):
            stats = svc.compact_if_needed(cid)
        self.assertEqual(stats["compacted"], 0)
        self.assertEqual(stats["forgot"], 0)
        kept = EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
            cid, limit=1000
        )
        self.assertEqual(len(kept), 21)
        self.assertEqual([(r.turn_start, r.turn_end) for r in kept[:2]], oldest)

    def test_merge_abort_grown_tokens_keeps_originals(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        repo = EpisodicMemoryRepository(self.db)
        repo.create(cid, "aa", 0, 8)
        repo.create(cid, "bb", 8, 16)
        for i in range(2, 21):
            repo.create(cid, f"summary number {i} extra words here", i * 8, (i + 1) * 8)
        for r in repo.list_ordered_for_compaction(cid, limit=1000)[:-10]:
            r.recall_count = 1
        self.db.commit()
        svc = EpisodicService(self.db)
        big = "word " * 500
        with patch.object(sum_mod, "summarize_text", return_value=big):
            stats = svc.compact_if_needed(cid)
        self.assertEqual(stats["compacted"], 0)
        self.assertEqual(stats["forgot"], 0)
        rows = repo.list_ordered_for_compaction(cid, limit=1000)
        self.assertEqual(len(rows), 21)

    def test_newest_three_never_merged(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        self.seed_rows(cid, 21)
        before = {
            r.summary
            for r in EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
                cid, limit=1000
            )[-3:]
        }
        with patch.object(sum_mod, "summarize_text", return_value="merged ab"):
            EpisodicService(self.db).compact_if_needed(cid)
        after = [
            r.summary
            for r in EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
                cid, limit=1000
            )
        ]
        for s in before:
            self.assertIn(s, after)

    def test_forget_zero_hit_oldest_first(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        rows = self.seed_rows(cid, 22)
        # Mark the third-oldest as hit so forget must skip it.
        hit = rows[2]
        hit.recall_count = 1
        self.db.commit()
        victim_summary = rows[3].summary
        hit_summary = hit.summary
        with patch.object(sum_mod, "summarize_text", return_value="merged ab"):
            stats = EpisodicService(self.db).compact_if_needed(cid)
        self.assertEqual(stats["compacted"], 1)
        self.assertEqual(stats["forgot"], 1)
        remaining = [
            r.summary
            for r in EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
                cid, limit=1000
            )
        ]
        self.assertIn(hit_summary, remaining)
        self.assertNotIn(victim_summary, remaining)
        self.assertEqual(len(remaining), 20)

    def test_forget_never_touches_newest_ten(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        rows = self.seed_rows(cid, 22)
        # Every row outside the newest-10 is a hit, so nothing is forgettable.
        for r in rows[:-10]:
            r.recall_count = 1
        self.db.commit()
        newest_ten = {
            r.summary for r in rows[-10:]
        }
        with patch.object(sum_mod, "summarize_text", return_value="merged ab"):
            stats = EpisodicService(self.db).compact_if_needed(cid)
        self.assertEqual(stats["forgot"], 0)
        remaining = {
            r.summary
            for r in EpisodicMemoryRepository(self.db).list_ordered_for_compaction(
                cid, limit=1000
            )
        }
        # Newest rows survive (minus at most the merge pair, which is oldest).
        for s in newest_ten:
            self.assertIn(s, remaining)

    def test_recall_increments_count_and_stamps_time(self):
        cid = self.conv()
        repo = EpisodicMemoryRepository(self.db)
        repo.create(cid, "user likes ramen noodles", 0, 8)
        svc = EpisodicService(self.db)
        out = svc.recall(cid, "what ramen to cook?", limit=3)
        self.assertTrue(out[0].startswith("user likes ramen"))
        row = repo.list_by_conversation(cid)[0]
        self.assertEqual(int(row.recall_count or 0), 1)
        self.assertIsNotNone(row.last_recalled_at)
        svc.recall(cid, "ramen again", limit=3)
        row = repo.list_by_conversation(cid)[0]
        self.assertEqual(int(row.recall_count or 0), 2)

    def test_recall_vector_hit_increments(self):
        cid = self.conv()
        repo = EpisodicMemoryRepository(self.db)
        coffee = repo.create(cid, "user prefers strong coffee", 0, 4)
        repo.create(cid, "project deadline is friday", 4, 8)
        svc = EpisodicService(self.db)
        out = svc.recall(
            cid,
            "morning brew habit",
            limit=3,
            query_vector=[1.0, 0.0],
            candidate_vectors={str(coffee.id): [1.0, 0.0]},
        )
        self.assertEqual(out, ["user prefers strong coffee"])
        row = repo.get_by_id(coffee.id)
        self.assertEqual(int(row.recall_count or 0), 1)
        self.assertIsNotNone(row.last_recalled_at)

    def test_recall_ordering_preserved_post_merge(self):
        import services.summarize_service as sum_mod

        cid = self.conv()
        repo = EpisodicMemoryRepository(self.db)
        # Oldest pair is filler so the merge never touches recall targets.
        for i in range(19):
            repo.create(cid, f"filler note number {i} extra words", i * 8, (i + 1) * 8)
        repo.create(cid, "project deadline is friday", 19 * 8, 20 * 8)
        repo.create(cid, "user likes ramen noodles", 20 * 8, 21 * 8)
        with patch.object(sum_mod, "summarize_text", return_value="merged ab"):
            EpisodicService(self.db).compact_if_needed(cid)
        out = EpisodicService(self.db).recall(cid, "what ramen to cook?", limit=3)
        self.assertTrue(out[0].startswith("user likes ramen"))


if __name__ == "__main__":
    unittest.main()
