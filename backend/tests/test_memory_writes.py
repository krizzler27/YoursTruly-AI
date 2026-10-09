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

    def test_tag_request_round_trip(self):
        from router.chat_api import update_conversation
        from schemas.api_schemas import ConversationUpdateRequest
        from services.chat_services import ChatServices

        svc = ChatServices(self.db)
        conv = svc.ensure_conversation(None, title="Original")
        tagged = update_conversation(
            conv.id, ConversationUpdateRequest.model_validate({"tag": "laya"}), self.db
        )
        self.assertEqual(tagged.tag, "laya")
        self.assertEqual(tagged.title, "Original")

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


class ProjectAggregation(DbCase):
    def seed_messages(self, cid, n=10, text="I enjoy late night coding sessions"):
        from db.models import MessagesModel

        for i in range(n):
            self.db.add(
                MessagesModel(
                    conversation_id=cid,
                    role="user" if i % 2 == 0 else "assistant",
                    content=text,
                )
            )
        self.db.commit()

    def test_rollup_updates_project_summary(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod
        from repository.episodic_repository import EpisodicMemoryRepository
        from repository.project_repository import ProjectRepository
        from services import semantic_writer
        from services.chat_services import ChatServices

        svc = ChatServices(self.db)
        proj = ProjectRepository(self.db).get_or_create("laya")
        c1 = svc.ensure_conversation(None, title="one").id
        c2 = svc.ensure_conversation(None, title="two").id
        svc.set_project(c1, proj.id)
        svc.set_project(c2, proj.id)
        EpisodicMemoryRepository(self.db).create(c1, "decided launch friday", 0, 4)
        EpisodicMemoryRepository(self.db).create(c2, "picked oat milk", 0, 4)
        self.seed_messages(c1)
        idle = MagicMock()
        idle.is_generating.return_value = False
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", return_value="rolled"),
            patch.object(semantic_writer, "extract_facts_slm", return_value=[]),
        ):
            rollup_mod.rollup_job.run((str(c1), "test-req"))
        self.assertEqual(ProjectRepository(self.db).get_by_name("laya").summary, "rolled")

    def test_unlinked_chat_leaves_projects_untouched(self):
        import services.rollup_job as rollup_mod
        import services.summarize_service as sum_mod
        from repository.project_repository import ProjectRepository
        from services import semantic_writer
        from services.chat_services import ChatServices

        cid = ChatServices(self.db).ensure_conversation(None, title="solo").id
        self.seed_messages(cid)
        idle = MagicMock()
        idle.is_generating.return_value = False
        with (
            patch.object(rollup_mod, "SessionLocal", lambda: self.db),
            patch("services.llama_engine.LlamaEngine.get_instance", return_value=idle),
            patch.object(sum_mod, "summarize_text", return_value="rolled"),
            patch.object(semantic_writer, "extract_facts_slm", return_value=[]),
        ):
            rollup_mod.rollup_job.run((str(cid), "test-req"))
        self.assertEqual(ProjectRepository(self.db).list_all(), [])

    def test_member_chat_reads_project_summary(self):
        from unittest.mock import MagicMock as MM

        from repository.project_repository import ProjectRepository
        from schemas.rag_schemas import RouteDecision
        from services.chat_services import ChatServices
        from services.rag_graph import RagGraph
        from services.rag_service import RagService

        proj = ProjectRepository(self.db).get_or_create("laya")
        ProjectRepository(self.db).update_summary(proj.id, "launch friday")
        member = ChatServices(self.db).ensure_conversation(None, title="m").id
        ChatServices(self.db).set_project(member, proj.id)
        outsider = ChatServices(self.db).ensure_conversation(None, title="o").id

        class _FakeEmbed:
            def embed(self, texts):
                return [[0.0, 1.0] for _ in texts]

            def unload(self):
                pass

        def graph():
            rag = MM()
            rag.db = self.db
            rag.engine = _FakeEmbed()
            rag.build_messages.side_effect = (
                lambda q, h, hist, **kw: [
                    {"role": "u", "content": "\n".join(kw.get("topic_lines") or [])}
                ]
            )
            decider = MM()
            decider.decide.return_value = RouteDecision(route="DIRECT", reason="t")
            return RagGraph(db=self.db, rag=rag, decider=decider, llm=MM())

        with patch(
            "services.decider.LayaService.get_instance",
            side_effect=RuntimeError("laya down"),
        ):
            member_text = graph()._build(
                {"query": "hi", "history": [], "conversation_id": member}
            )["messages"][0]["content"]
            outsider_text = graph()._build(
                {"query": "hi", "history": [], "conversation_id": outsider}
            )["messages"][0]["content"]
        self.assertIn("launch friday", member_text)
        self.assertNotIn("launch friday", outsider_text)


class RecallPriority(DbCase):
    def member_with_project(self, summary="launch friday"):
        from repository.project_repository import ProjectRepository
        from services.chat_services import ChatServices

        proj = ProjectRepository(self.db).get_or_create("laya")
        ProjectRepository(self.db).update_summary(proj.id, summary)
        member = ChatServices(self.db).ensure_conversation(None, title="m").id
        ChatServices(self.db).set_project(member, proj.id)
        return member

    def test_project_rides_history_remainder_not_memory_carve(self):
        from unittest.mock import MagicMock as MM

        from schemas.rag_schemas import RouteDecision
        from services.rag_graph import RagGraph

        member = self.member_with_project()

        class _FakeEmbed:
            def embed(self, texts):
                return [[0.0, 1.0] for _ in texts]

            def unload(self):
                pass

        rag = MM()
        rag.db = self.db
        rag.engine = _FakeEmbed()
        g = RagGraph(db=self.db, rag=rag, decider=MM(), llm=MM())
        with patch(
            "services.decider.LayaService.get_instance",
            side_effect=RuntimeError("laya down"),
        ):
            mem_text = g._memory_text("launch", member) or ""
            lines = g._project_summary_line(member)
        self.assertNotIn("launch friday", mem_text)
        self.assertEqual(len(lines), 1)
        self.assertIn("launch friday", lines[0])

    def test_line_caps_hold(self):
        from unittest.mock import MagicMock as MM

        from repository.episodic_repository import EpisodicMemoryRepository
        from schemas.rag_schemas import RouteDecision
        from services.chat_services import ChatServices
        from services.rag_graph import RagGraph

        member = self.member_with_project()
        ChatServices(self.db).set_tag(member, "laya")
        for i in range(5):
            sib = ChatServices(self.db).ensure_conversation(None, title=f"s{i}").id
            ChatServices(self.db).set_tag(sib, "laya")
            EpisodicMemoryRepository(self.db).create(sib, f"sibling summary {i}", 0, 4)
        rag = MM()
        rag.db = self.db
        g = RagGraph(db=self.db, rag=rag, decider=MM(), llm=MM())
        lines = g._project_summary_line(member) + g._tag_sibling_lines(member)
        self.assertLessEqual(len(lines), 4)
        self.assertEqual(
            len([ln for ln in lines if ln.startswith("Earlier in project")]), 3
        )
        self.assertEqual(len([ln for ln in lines if ln.startswith("Project ")]), 1)
        self.assertTrue(lines[0].startswith("Project "))

    def test_all_four_lines_survive_with_room(self):
        from unittest.mock import MagicMock as MM

        from services.rag_service import RagService

        rag = RagService(db=self.db, engine=MM(), lance=MM(), docs=MM())
        lines = ["Project laya summary: launch friday"] + [
            f"Earlier in project laya: sibling summary {i}" for i in range(3)
        ]
        msgs = rag.build_messages("what did we decide?", None, [], topic_lines=lines)
        system = msgs[0]["content"]
        self.assertIn("launch friday", system)
        for i in range(3):
            self.assertIn(f"sibling summary {i}", system)

    def test_tight_remainder_keeps_project_over_siblings(self):
        from services.rag_service import _fit_topic

        lines = ["Project laya summary: launch friday"] + [
            f"Earlier in project laya: sibling summary {i}" for i in range(3)
        ]
        block, _ = _fit_topic(lines, 12)
        self.assertIn("launch friday", block)
        self.assertNotIn("sibling summary 2", block)

    def test_own_history_squeezes_project_lines(self):
        from unittest.mock import MagicMock as MM

        from core.context_budget import allocate as real_allocate
        from services.rag_service import RagService

        member = self.member_with_project()
        history = [
            {"role": "user", "content": "alpha " * 40},
            {"role": "assistant", "content": "beta " * 40},
        ]
        rag = RagService(db=self.db, engine=MM(), lance=MM(), docs=MM())
        full = rag.build_messages(
            "what did we decide?",
            None,
            history,
            topic_lines=["Project laya summary: launch friday"],
        )
        self.assertIn("launch friday", full[0]["content"])

        def tiny_allocate(*a, **k):
            budget = real_allocate(*a, **k)
            budget["history_cap"] = 4
            return budget

        with patch("services.rag_service.allocate", side_effect=tiny_allocate):
            starved = rag.build_messages(
                "what did we decide?",
                None,
                history,
                topic_lines=["Project laya summary: launch friday"],
            )
        system = starved[0]["content"]
        self.assertNotIn("launch friday", system)
        self.assertIn("beta", system)


if __name__ == "__main__":
    unittest.main()
