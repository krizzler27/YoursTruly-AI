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

        source = inspect.getsource(semantic_writer).lower()
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


if __name__ == "__main__":
    unittest.main()
