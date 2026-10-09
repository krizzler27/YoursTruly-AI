"""Persisted vector recall: write-time embed, stored cosine, single query embed."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from db.models import Base


class FakeCoffeeEmbed:
    """Coffee prefer axis versus everything else, records call sizes."""

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
            if "coffee" in low or "prefer" in low:
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out

    def unload(self):
        self.unloaded = getattr(self, "unloaded", 0) + 1

    def is_loaded(self):
        return True

    def ensure_loaded(self):
        return None


class ColdEngine(FakeCoffeeEmbed):
    """Embed engine present but weights unloaded."""

    def is_loaded(self):
        return False


class DbCase(unittest.TestCase):
    def setUp(self):
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()

    def tearDown(self):
        self.db.close()


def _repo(db):
    from repository.semantic_repository import SemanticMemoryRepository

    return SemanticMemoryRepository(db)


class StoredVectorRecall(DbCase):
    def test_paraphrase_hits_without_batch_embed(self):
        r = _repo(self.db)
        r.upsert("preference", "strong coffee", embedding=[1.0, 0.0])
        r.upsert("name", "Ada", embedding=[0.0, 1.0])
        q = "what coffee do I prefer"
        self.assertEqual(r.find_relevant(q, limit=5), [])
        hits = r.find_relevant(q, limit=5, query_vector=[1.0, 0.0])
        self.assertEqual([h.key for h in hits], ["preference"])

    def test_overlap_keeps_priority_and_cap_holds(self):
        r = _repo(self.db)
        r.upsert("name", "Ada", embedding=[0.0, 1.0])
        r.upsert("preference", "strong coffee", embedding=[0.0, 1.0])
        hits = r.find_relevant(
            "what is my name?", limit=5, query_vector=[0.0, 1.0]
        )
        self.assertEqual([h.key for h in hits][0], "name")
        capped = r.find_relevant(
            "what is my name?", limit=1, query_vector=[0.0, 1.0]
        )
        self.assertEqual(len(capped), 1)

    def test_threshold_blocks_weak_cosine(self):
        r = _repo(self.db)
        r.upsert("preference", "strong coffee", embedding=[0.0, 1.0])
        hits = r.find_relevant(
            "what coffee do I prefer", limit=5, query_vector=[1.0, 0.0]
        )
        self.assertEqual(hits, [])

    def test_candidate_override_still_works(self):
        r = _repo(self.db)
        r.upsert("name", "Ada", embedding=[0.0, 1.0])
        hits = r.find_relevant(
            "how should I address you",
            limit=5,
            query_vector=[1.0, 0.0],
            candidate_vectors={"name": [1.0, 0.0]},
        )
        self.assertEqual([h.key for h in hits], ["name"])


class WriteTimeVector(DbCase):
    def test_remember_stores_vector_at_write_time(self):
        from services.semantic_writer import remember_explicit

        fake = FakeCoffeeEmbed()
        with patch(
            "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
        ):
            row = remember_explicit(self.db, "I prefer strong coffee")
        self.assertIsNotNone(row)
        self.assertEqual(row.key, "preference")
        from repository.semantic_repository import decode_embedding

        vec = decode_embedding(getattr(row, "embedding", None))
        self.assertIsNotNone(vec)
        self.assertEqual(len(vec), 2)
        hits = _repo(self.db).find_relevant(
            "what coffee do I prefer", limit=5, query_vector=[1.0, 0.0]
        )
        self.assertEqual([h.key for h in hits], ["preference"])

    def test_foreground_loads_cold_engine_on_demand(self):
        from repository.semantic_repository import decode_embedding
        from services.semantic_writer import remember_explicit

        fake = ColdEngine()
        with patch(
            "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
        ):
            row = remember_explicit(self.db, "I prefer strong coffee")
        self.assertIsNotNone(row)
        self.assertEqual(len(fake.calls), 1)
        self.assertGreaterEqual(getattr(fake, "unloaded", 0), 1)
        self.assertIsNotNone(decode_embedding(getattr(row, "embedding", None)))


class EpisodicVectors(DbCase):
    def test_rollup_stores_summary_vector(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod
        from db.models import MessagesModel
        from repository.episodic_repository import EpisodicMemoryRepository
        from repository.semantic_repository import decode_embedding
        from services import semantic_writer
        from services.chat_services import ChatServices

        cid = ChatServices(self.db).ensure_conversation(None, title="t").id
        for i in range(10):
            self.db.add(
                MessagesModel(
                    conversation_id=cid,
                    role="user" if i % 2 == 0 else "assistant",
                    content="we decided the launch plan last tuesday",
                )
            )
        self.db.commit()
        idle = MagicMock()
        idle.is_generating.return_value = False
        fake = FakeCoffeeEmbed()
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", return_value="decided launch friday"),
            patch.object(semantic_writer, "extract_facts_slm", return_value=[]),
            patch(
                "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
            ),
        ):
            rollup_mod.rollup_job.run((str(cid), "test-req"))
        rows = EpisodicMemoryRepository(self.db).list_by_conversation(cid)
        self.assertTrue(rows)
        self.assertIsNotNone(decode_embedding(getattr(rows[0], "embedding", None)))

    def test_recall_uses_stored_vectors_without_batch(self):
        from repository.episodic_repository import EpisodicMemoryRepository
        from services.episodic_service import EpisodicService

        cid = self.conv_id()
        EpisodicMemoryRepository(self.db).create(
            cid, "we decide the launch plan last tuesday", 0, 4, embedding=[1.0, 0.0]
        )
        texts = EpisodicService(self.db).recall(
            cid, "when is the bursting supernova gala", limit=3,
            query_vector=[1.0, 0.0],
        )
        self.assertEqual(texts, ["we decide the launch plan last tuesday"])

    def test_backfill_heals_vector_less_rows(self):
        from repository.episodic_repository import EpisodicMemoryRepository
        from repository.semantic_repository import SemanticMemoryRepository, decode_embedding
        from services.semantic_writer import backfill_missing_vectors

        SemanticMemoryRepository(self.db).upsert("hobby", "late night coding")
        cid = self.conv_id()
        EpisodicMemoryRepository(self.db).create(cid, "decided launch friday", 0, 4)
        fake = FakeCoffeeEmbed()
        with patch(
            "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
        ):
            healed = backfill_missing_vectors(self.db, limit=10)
        self.assertEqual(healed, 2)
        self.assertIsNotNone(
            decode_embedding(SemanticMemoryRepository(self.db).get_by_key("hobby").embedding)
        )
        rows = EpisodicMemoryRepository(self.db).list_by_conversation(cid)
        self.assertIsNotNone(decode_embedding(getattr(rows[0], "embedding", None)))

    def test_backfill_respects_cap(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services.semantic_writer import backfill_missing_vectors

        for i in range(5):
            SemanticMemoryRepository(self.db).upsert(f"fact {i}", f"value {i}")
        fake = FakeCoffeeEmbed()
        with patch(
            "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
        ):
            healed = backfill_missing_vectors(self.db, limit=2)
        self.assertEqual(healed, 2)

    def conv_id(self):
        from db.models import ConversationsModel

        row = ConversationsModel(title="t")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row.id

    def test_remember_fail_open_stores_row_with_empty_vector(self):
        from services.semantic_writer import remember_explicit

        fake = FakeCoffeeEmbed(fail=True)
        with patch(
            "services.llama_engine.EmbeddingEngine.get_instance", return_value=fake
        ):
            row = remember_explicit(self.db, "I prefer strong coffee")
        self.assertIsNotNone(row)
        self.assertEqual(row.value, "strong coffee")
        raw = getattr(row, "embedding", None)
        self.assertTrue(raw is None or raw == "" or raw == "[]")

    def test_extract_and_store_persists_vectors(self):
        from services import semantic_writer

        msgs = [{"role": "user", "content": "just noting tonight"}]
        fake = FakeCoffeeEmbed()
        with (
            patch.object(
                semantic_writer,
                "extract_facts_slm",
                return_value=[("preference", "strong coffee")],
            ),
            patch(
                "services.llama_engine.EmbeddingEngine.get_instance",
                return_value=fake,
            ),
        ):
            stored = semantic_writer.extract_and_store(self.db, msgs)
        self.assertEqual(stored, 1)
        from repository.semantic_repository import decode_embedding

        row = _repo(self.db).get_by_key("preference")
        self.assertIsNotNone(decode_embedding(getattr(row, "embedding", None)))


OLD_SEMANTIC = """
CREATE TABLE semantic_memory (
    id CHAR(32) NOT NULL,
    key VARCHAR NOT NULL,
    value VARCHAR NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (id)
)
"""


class VectorSchema(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "drifted.db")
        self.eng = create_engine(f"sqlite:///{self.path}")

    def tearDown(self):
        self.eng.dispose()

    def test_old_db_gains_embedding_column_rows_kept(self):
        from db.migrate import ensure_schema

        with self.eng.begin() as conn:
            conn.execute(text(OLD_SEMANTIC))
        before = {c["name"] for c in inspect(self.eng).get_columns("semantic_memory")}
        self.assertNotIn("embedding", before)
        with self.eng.begin() as conn:
            conn.execute(
                text("INSERT INTO semantic_memory (id, key, value) VALUES (:id, :k, :v)"),
                {"id": "abc123", "k": "preference", "v": "strong coffee"},
            )
        added = ensure_schema(self.eng)
        self.assertIn("semantic_memory.embedding", added)
        after = {c["name"] for c in inspect(self.eng).get_columns("semantic_memory")}
        self.assertIn("embedding", after)
        with self.eng.connect() as conn:
            val = conn.execute(
                text("SELECT value FROM semantic_memory WHERE key = 'preference'")
            ).scalar()
        self.assertEqual(val, "strong coffee")
        self.assertEqual(ensure_schema(self.eng), [])

    def test_fresh_create_all_has_embedding(self):
        self.eng.dispose()
        mem = create_engine("sqlite:///:memory:")
        try:
            Base.metadata.create_all(mem)
            cols = {c["name"] for c in inspect(mem).get_columns("semantic_memory")}
            self.assertIn("embedding", cols)
        finally:
            mem.dispose()


class GraphStoredRecall(DbCase):
    def memory_graph(self, engine):
        from services.rag_graph import RagGraph

        rag = MagicMock()
        rag.db = self.db
        rag.engine = engine
        rag.build_messages.side_effect = lambda q, h, hist, **kw: [
            {"role": "u", "content": kw.get("memory_text") or ""}
        ]
        decider = MagicMock()
        decider.decide.return_value = MagicMock(route="DIRECT", reason="t")
        return RagGraph(db=self.db, rag=rag, decider=decider, llm=MagicMock()), rag

    def test_paraphrase_recall_with_single_query_embed(self):
        from unittest.mock import patch as mock_patch

        r = _repo(self.db)
        r.upsert("preference", "strong coffee", embedding=[1.0, 0.0])
        engine = FakeCoffeeEmbed()
        g, _ = self.memory_graph(engine)
        with mock_patch(
            "services.decider.LayaService.get_instance",
            side_effect=RuntimeError("laya down"),
        ):
            txt = g._memory_text("what coffee do I prefer")
        self.assertIsNotNone(txt)
        self.assertIn("preference: strong coffee", txt)
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(len(engine.calls[0]), 1)

    def test_embed_down_fails_open_to_overlap_only(self):
        from unittest.mock import patch as mock_patch

        r = _repo(self.db)
        r.upsert("preference", "strong coffee", embedding=[1.0, 0.0])
        engine = FakeCoffeeEmbed(fail=True)
        g, _ = self.memory_graph(engine)
        with mock_patch(
            "services.decider.LayaService.get_instance",
            side_effect=RuntimeError("laya down"),
        ):
            self.assertIsNone(g._memory_text("what coffee do I prefer"))
            r2 = _repo(self.db)
            r2.upsert("name", "Ada", embedding=[1.0, 0.0])
            txt = g._memory_text("what is my name?")
        self.assertIn("Ada", txt or "")


if __name__ == "__main__":
    unittest.main()
