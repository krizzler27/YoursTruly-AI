"""Migration tests: drifted old-schema DBs gain new columns without data loss.

Runs offline (no GGUF): temp-file SQLite only, deleted after.
Never touches the real yourstrulyai.db (AGENTS testing rule).
"""

import sys
import tempfile
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from db.migrate import ensure_schema
from db.models import (
    Base,
    ConversationsModel,
    EpisodicMemoryModel,
)

OLD_CONVERSATIONS = """
CREATE TABLE conversations (
    id CHAR(32) NOT NULL,
    title VARCHAR NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (id)
)
"""

OLD_EPISODIC = """
CREATE TABLE episodic_memory (
    id CHAR(32) NOT NULL,
    conversation_id CHAR(32) NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    summary VARCHAR NOT NULL,
    turn_start INTEGER NOT NULL,
    turn_end INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY(conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
)
"""


def _columns(engine, table):
    return {c["name"]: c for c in inspect(engine).get_columns(table)}


class MigrateCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = str(Path(self.tmp.name) / "drifted.db")
        self.eng = create_engine(f"sqlite:///{self.path}")

    def tearDown(self):
        self.eng.dispose()

    def make_drifted(self):
        with self.eng.begin() as conn:
            conn.execute(text(OLD_CONVERSATIONS))
            conn.execute(text(OLD_EPISODIC))
        self.conv_id = uuid.uuid4().hex
        self.epi_id = uuid.uuid4().hex
        with self.eng.begin() as conn:
            conn.execute(
                text("INSERT INTO conversations (id, title) VALUES (:id, :title)"),
                {"id": self.conv_id, "title": "old chat"},
            )
            conn.execute(
                text(
                    "INSERT INTO episodic_memory "
                    "(id, conversation_id, summary, turn_start, turn_end) "
                    "VALUES (:id, :cid, :summary, 0, 8)"
                ),
                {"id": self.epi_id, "cid": self.conv_id, "summary": "old summary"},
            )


class DriftedDb(MigrateCase):
    def test_missing_columns_added_rows_kept(self):
        self.make_drifted()
        before = _columns(self.eng, "conversations")
        self.assertNotIn("tag", before)
        self.assertNotIn("project_id", before)
        epi_before = _columns(self.eng, "episodic_memory")
        self.assertNotIn("recall_count", epi_before)
        self.assertNotIn("last_recalled_at", epi_before)

        added = ensure_schema(self.eng)

        self.assertEqual(
            set(added),
            {
                "conversations.tag",
                "conversations.project_id",
                "episodic_memory.recall_count",
                "episodic_memory.last_recalled_at",
                "episodic_memory.embedding",
            },
        )
        after = _columns(self.eng, "conversations")
        self.assertIn("tag", after)
        self.assertIn("VARCHAR(64)", str(after["tag"]["type"]))
        self.assertIn("project_id", after)
        epi_after = _columns(self.eng, "episodic_memory")
        self.assertIn("recall_count", epi_after)
        self.assertIn("INTEGER", str(epi_after["recall_count"]["type"]))
        self.assertEqual(str(epi_after["recall_count"]["default"]).strip(), "0")
        self.assertIn("last_recalled_at", epi_after)
        self.assertIn("DATETIME", str(epi_after["last_recalled_at"]["type"]))

        with self.eng.connect() as conn:
            title = conn.execute(
                text("SELECT title FROM conversations WHERE id = :id"),
                {"id": self.conv_id},
            ).scalar()
            summary = conn.execute(
                text("SELECT summary FROM episodic_memory WHERE id = :id"),
                {"id": self.epi_id},
            ).scalar()
            recall = conn.execute(
                text("SELECT recall_count FROM episodic_memory WHERE id = :id"),
                {"id": self.epi_id},
            ).scalar()
        self.assertEqual(title, "old chat")
        self.assertEqual(summary, "old summary")
        self.assertEqual(int(recall), 0)

    def test_tag_index_created(self):
        self.make_drifted()
        ensure_schema(self.eng)
        idx_names = {i["name"] for i in inspect(self.eng).get_indexes("conversations")}
        self.assertIn("ix_conversations_tag", idx_names)

    def test_app_level_insert_with_tag_works(self):
        self.make_drifted()
        ensure_schema(self.eng)
        db = sessionmaker(bind=self.eng)()
        try:
            row = ConversationsModel(title="new chat", tag="laya")
            db.add(row)
            db.commit()
            db.refresh(row)
            self.assertEqual(row.tag, "laya")
            epi = EpisodicMemoryModel(
                conversation_id=row.id, summary="fresh note",
                turn_start=0, turn_end=8,
            )
            db.add(epi)
            db.commit()
            db.refresh(epi)
            self.assertEqual(int(epi.recall_count or 0), 0)
            self.assertIsNone(epi.last_recalled_at)
        finally:
            db.close()

    def test_second_run_is_noop(self):
        self.make_drifted()
        first = ensure_schema(self.eng)
        self.assertEqual(len(first), 5)
        second = ensure_schema(self.eng)
        self.assertEqual(second, [])


class FreshDb(MigrateCase):
    def test_fresh_create_all_then_migrate_is_noop(self):
        Base.metadata.create_all(self.eng)
        added = ensure_schema(self.eng)
        self.assertEqual(added, [])
        db = sessionmaker(bind=self.eng)()
        try:
            row = ConversationsModel(title="t", tag="laya")
            db.add(row)
            db.commit()
            db.refresh(row)
            self.assertEqual(row.tag, "laya")
        finally:
            db.close()
        # Idempotent on the fresh path too.
        self.assertEqual(ensure_schema(self.eng), [])

    def test_empty_db_without_tables_is_noop(self):
        self.assertEqual(ensure_schema(self.eng), [])


if __name__ == "__main__":
    unittest.main()
