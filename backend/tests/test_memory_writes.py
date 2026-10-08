"""MW2 explicit-remember write path: hardcoded no-model facts."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.models import Base, ConversationsModel


class DbCase(unittest.TestCase):
    def setUp(self):
        self.eng = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.eng)
        self.db = sessionmaker(bind=self.eng)()

    def tearDown(self):
        self.db.close()


class ParseExplicitFact(unittest.TestCase):
    def parse(self, text):
        from services.semantic_writer import parse_explicit_fact

        return parse_explicit_fact(text)

    def test_name_patterns(self):
        self.assertEqual(self.parse("please remember my name is Ada"), ("name", "Ada"))
        self.assertEqual(self.parse("call me Ash"), ("name", "Ash"))
        self.assertEqual(self.parse("my name is Ada"), ("name", "Ada"))

    def test_preference_patterns(self):
        key, value = self.parse("remember I prefer strong coffee")
        self.assertIn("strong coffee", value)
        key2, value2 = self.parse("I like strong coffee")
        self.assertIn("strong coffee", value2)

    def test_generic_remember(self):
        parsed = self.parse("remember the garage code is 4410")
        self.assertIsNotNone(parsed)
        key, value = parsed
        self.assertIn("4410", value)

    def test_questions_do_not_parse(self):
        self.assertIsNone(self.parse("how are you today"))
        self.assertIsNone(self.parse("what is my name?"))
        self.assertIsNone(self.parse("   "))
        self.assertIsNone(self.parse("do you remember my name?"))
        self.assertIsNone(self.parse("my dog is sick, what should I do?"))

    def test_generic_fact_key_supports_correction(self):
        first = self.parse("remember the garage code is 4410")
        second = self.parse("remember the garage code is 4420")
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(first[0], second[0])
        self.assertIn("4420", second[1])

    def test_name_value_stays_short(self):
        self.assertEqual(self.parse("call me Ash tomorrow"), ("name", "Ash"))
        self.assertEqual(
            self.parse("my name is Ada and I like coffee"), ("name", "Ada")
        )


class RememberExplicit(DbCase):
    def test_store_and_recall_next_chat(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services.semantic_writer import remember_explicit

        row = remember_explicit(self.db, "my name is Ada")
        self.assertIsNotNone(row)
        self.assertEqual(
            SemanticMemoryRepository(self.db).get_by_key("name").value, "Ada"
        )
        # next chat recalls inside the memory block
        hits = SemanticMemoryRepository(self.db).find_relevant("what is my name?")
        self.assertEqual([h.key for h in hits], ["name"])

    def test_overwrite_on_repeat(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services.semantic_writer import remember_explicit

        remember_explicit(self.db, "my name is Ada")
        remember_explicit(self.db, "call me Grace")
        repo = SemanticMemoryRepository(self.db)
        self.assertEqual(repo.get_by_key("name").value, "Grace")
        self.assertEqual(len(repo.list_all(limit=100)), 1)

    def test_non_fact_returns_none_and_stores_nothing(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services.semantic_writer import remember_explicit

        self.assertIsNone(remember_explicit(self.db, "how are you today"))
        self.assertEqual(SemanticMemoryRepository(self.db).list_all(limit=100), [])

    def test_fail_open_on_db_error(self):
        from services.semantic_writer import remember_explicit

        bad = MagicMock()
        bad.query.side_effect = RuntimeError("db down")
        self.assertIsNone(remember_explicit(bad, "my name is Ada"))

    def test_no_model_call_on_this_path(self):
        import inspect

        from services import semantic_writer

        source = (
            inspect.getsource(semantic_writer.parse_explicit_fact)
            + inspect.getsource(semantic_writer.remember_explicit)
        ).lower()
        for token in ("llmservice", "llamaengine", "workerengine", "invoke("):
            self.assertNotIn(token, source)

    def test_chat_service_wrapper_is_fail_open(self):
        from services.chat_services import ChatServices

        svc = ChatServices(self.db)
        row = svc.maybe_remember("my name is Ada")
        self.assertIsNotNone(row)
        self.assertIsNone(svc.maybe_remember("how are you today"))
        with patch(
            "services.semantic_writer.remember_explicit",
            side_effect=RuntimeError("boom"),
        ):
            self.assertIsNone(svc.maybe_remember("my name is Ada"))


class ExtractAndStore(DbCase):
    def test_slm_facts_store_with_mw2_normalization(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services import semantic_writer

        msgs = [
            {"role": "user", "content": "I like strong coffee"},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "how are you today"},
        ]
        with patch.object(
            semantic_writer,
            "extract_facts_slm",
            return_value=[(" Favorite Drink ", " Espresso ")],
        ):
            stored = semantic_writer.extract_and_store(self.db, msgs)
        self.assertEqual(stored, 1)
        repo = SemanticMemoryRepository(self.db)
        self.assertEqual(repo.get_by_key("favorite drink").value, "Espresso")
        self.assertEqual(
            [h.key for h in repo.find_relevant("favorite drink?")],
            ["favorite drink"],
        )

    def test_busy_timeout_empty_skip_fail_open(self):
        from repository.semantic_repository import SemanticMemoryRepository
        from services import semantic_writer

        msgs = [{"role": "user", "content": "I like strong coffee"}]
        for failure in (
            RuntimeError("System Busy - model is generating. Try again."),
            TimeoutError("slow"),
        ):
            with patch.object(
                semantic_writer, "extract_facts_slm", side_effect=failure
            ):
                self.assertEqual(semantic_writer.extract_and_store(self.db, msgs), 0)
        with patch.object(semantic_writer, "extract_facts_slm", return_value=[]):
            self.assertEqual(semantic_writer.extract_and_store(self.db, msgs), 0)
        self.assertEqual(SemanticMemoryRepository(self.db).list_all(limit=100), [])

    def test_rollup_worker_run_extracts(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod
        from db.models import MessagesModel
        from repository.semantic_repository import SemanticMemoryRepository
        from services import semantic_writer

        conv = ConversationsModel(title="t")
        self.db.add(conv)
        self.db.commit()
        for i in range(10):
            self.db.add(
                MessagesModel(
                    conversation_id=conv.id,
                    role="user" if i % 2 == 0 else "assistant",
                    content="I enjoy late night coding sessions",
                )
            )
        self.db.commit()
        idle = MagicMock()
        idle.is_generating.return_value = False
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", return_value="rolled"),
            patch.object(
                semantic_writer,
                "extract_facts_slm",
                return_value=[("hobby", "late night coding")],
            ),
        ):
            rollup_mod.rollup_job.run((str(conv.id), "test-req"))
        self.assertEqual(
            SemanticMemoryRepository(self.db).get_by_key("hobby").value,
            "late night coding",
        )


class TagRename(DbCase):
    def test_tag_crud_and_sibling_list(self):
        from repository.chat_repository import ChatRepository

        repo = ChatRepository(self.db)
        c1, c2, c3 = self.conv(), self.conv(), self.conv()
        repo.set_tag(c1, "laya")
        repo.set_tag(c2, "laya")
        repo.set_tag(c3, "other")
        self.assertEqual(repo.get_by_id(c1).tag, "laya")
        self.assertEqual([r.id for r in repo.list_by_tag("laya", exclude_id=c1)], [c2])
        repo.set_tag(c1, None)
        self.assertIsNone(repo.get_by_id(c1).tag)

    def test_topic_aliases_delegate_to_tag(self):
        from repository.chat_repository import ChatRepository

        repo = ChatRepository(self.db)
        cid = self.conv()
        repo.set_topic(cid, "laya")
        self.assertEqual(repo.get_by_id(cid).tag, "laya")
        self.assertEqual(repo.get_by_id(cid).topic, "laya")
        sib = self.conv()
        repo.set_tag(sib, "laya")
        self.assertEqual(
            {r.id for r in repo.list_by_topic("laya")}, {cid, sib}
        )

    def test_topic_request_field_maps_to_tag(self):
        from router.chat_api import update_conversation
        from schemas.api_schemas import ConversationUpdateRequest
        from services.chat_services import ChatServices

        svc = ChatServices(self.db)
        conv = svc.ensure_conversation(None, title="Original")
        tagged = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"topic": "laya"}), self.db
        )
        self.assertEqual(tagged.tag, "laya")
        self.assertEqual(tagged.title, "Original")
        renamed = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"tag": "other"}), self.db
        )
        self.assertEqual(renamed.tag, "other")

    def conv(self):
        row = ConversationsModel(title="t")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row.id


class ProjectLink(DbCase):
    def test_project_repo_crud(self):
        from repository.project_repository import ProjectRepository

        repo = ProjectRepository(self.db)
        proj = repo.get_or_create("laya")
        self.assertEqual(proj.name, "laya")
        self.assertEqual(repo.get_or_create("laya").id, proj.id)
        repo.update_summary(proj.id, "launch friday")
        self.assertEqual(repo.get_by_name("laya").summary, "launch friday")

    def test_conversation_links_to_project(self):
        from repository.chat_repository import ChatRepository
        from repository.project_repository import ProjectRepository
        from router.chat_api import update_conversation
        from schemas.api_schemas import ConversationUpdateRequest
        from services.chat_services import ChatServices

        proj = ProjectRepository(self.db).get_or_create("laya")
        conv = ChatServices(self.db).ensure_conversation(None, title="t")
        self.assertIsNone(conv.project_id)
        linked = ChatServices(self.db).set_project(conv.id, proj.id)
        self.assertEqual(linked.project_id, proj.id)
        self.assertEqual(ChatRepository(self.db).get_by_id(conv.id).project_id, proj.id)
        via_api = update_conversation(
            conv.id,
            ConversationUpdateRequest.model_validate({"project_id": str(proj.id)}),
            self.db,
        )
        self.assertEqual(via_api.project_id, proj.id)
        cleared = update_conversation(
            conv.id,
            ConversationUpdateRequest.model_validate({"project_id": None}),
            self.db,
        )
        self.assertIsNone(cleared.project_id)

    def conv(self):
        row = ConversationsModel(title="t")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row.id


class TopicRenameMigration(unittest.TestCase):
    def test_topic_values_survive_rename(self):
        import tempfile
        from pathlib import Path as FPath

        from sqlalchemy import create_engine, inspect, text

        from db.migrate import ensure_schema

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        eng = create_engine(f"sqlite:///{FPath(tmp.name) / 'old.db'}")
        self.addCleanup(eng.dispose)
        cid = "c" * 32
        with eng.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE conversations ("
                    "id CHAR(32) NOT NULL PRIMARY KEY, "
                    "title VARCHAR NOT NULL, "
                    "topic VARCHAR(64), "
                    "created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL, "
                    "updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL)"
                )
            )
            conn.execute(
                text("INSERT INTO conversations (id, title, topic) VALUES (:id, :t, :topic)"),
                {"id": cid, "t": "old chat", "topic": "laya"},
            )
        ensure_schema(eng)
        cols = {c["name"] for c in inspect(eng).get_columns("conversations")}
        self.assertIn("tag", cols)
        self.assertIn("project_id", cols)
        self.assertNotIn("topic", cols)
        with eng.connect() as conn:
            tag = conn.execute(
                text("SELECT tag FROM conversations WHERE id = :id"), {"id": cid}
            ).scalar()
            linked = conn.execute(
                text("SELECT project_id FROM conversations WHERE id = :id"), {"id": cid}
            ).scalar()
        self.assertEqual(tag, "laya")
        self.assertIsNone(linked)


if __name__ == "__main__":
    unittest.main()
