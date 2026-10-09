"""Agentic chat orchestration - decide, retrieve, build. Sole RAG entry point.

Flow: query + history + conversation_id -> decide (DIRECT skips retrieval)
-> retrieve (scoped hybrid search) -> build (grounded or plain messages).
"""

import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from langgraph.graph import END, StateGraph
from sqlalchemy.orm import Session

from schemas.rag_schemas import RouteDecision, SearchHit
from config import config
from core.context_budget import count_tokens, top_k_for_ctx
from core.logging import get_logger
from services.decider import Decider, memory_evidence
from repository.semantic_repository import SemanticMemoryRepository, memory_embed_text
from services.laya_service import LayaService
from services.llama_engine import EmbeddingEngine, LlamaEngine, WorkerEngine
from services.llm_service import LLMService
from services.prompt_manager import PromptManager
from services.rag_service import RagService, _fit_memory
from services.summarize_service import resolve_slot

logger = get_logger(__name__)


MEMORY_DOMINANT_SHARE = 0.7
TOPIC_SIBLING_MAX_LINES = 3
PROJECT_SUMMARY_MAX_LINES = 1

# Frozen rewrite-needed template, byte-identical to training rows and to
# the smoke test in backend/notebooks/laya_dataset/test_finetuned.py.
# Wording is model input: never reword without retraining (docs/laya.md 3.3).
REWRITE_QUESTION = {
    "type": "noul",
    "instructions": "Does this query need rewriting to be self-contained for retrieval?",
}

REWRITE_MAX_TOKENS = 64

# Own pool for the foreground chat slot only, mirroring summarize_text:
# timeout enforcement without a throwaway executor per call. Worker-slot
# calls never touch this pool; they block on their own owned slot.
_REWRITE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rewrite-chat")


def _rewrite_history_text(
    turns: Optional[List[Dict[str, str]]], cap_chars: int = 500
) -> str:
    """Last turns as User/Assistant lines, same shape as the decider gate."""
    items = list(turns or [])[-2:]
    if not items:
        return "No prior conversation."
    lines = []
    for m in items:
        role = "User" if (m or {}).get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(m or {}).get('content', '')}")
    text = "\n".join(lines).strip()
    if len(text) > cap_chars:
        text = text[:cap_chars].rstrip()
    return text or "No prior conversation."


def _run_rewrite_chat(prompt_text: str, max_tokens: int) -> str:
    """One SLM pass on the chat slot; Busy propagates to the caller."""
    engine = LlamaEngine.get_instance("chat")
    if engine.is_generating():
        raise RuntimeError("System Busy - model is generating. Try again.")
    engine.ensure_loaded()
    svc = LLMService(engine=engine)
    messages = [{"role": "user", "content": prompt_text}]
    return (svc.invoke(messages, max_tokens=max_tokens, temperature=0.2) or "").strip()


def _run_rewrite_worker(prompt_text: str, max_tokens: int) -> str:
    """One SLM pass on the worker slot; load on demand, unload after."""
    engine = WorkerEngine.get_instance("worker")
    engine.ensure_loaded()
    try:
        svc = LLMService(engine=engine)
        messages = [{"role": "user", "content": prompt_text}]
        return (svc.invoke(messages, max_tokens=max_tokens, temperature=0.2) or "").strip()
    finally:
        engine.unload()


def _rewrite_via_slm(
    query: str,
    history_text: str,
    filenames_block: str,
    timeout: Optional[float] = None,
    role: Optional[str] = None,
    max_tokens: int = REWRITE_MAX_TOKENS,
) -> str:
    """Expand a follow-up into a standalone query; never raises.

    Mirrors summarize_text slot discipline: role=None auto-routes via
    resolve_slot (worker preferred), the chat path busy-check-skips and
    enforces timeout through the shared pool, and every failure returns
    the original query. Call before any generation acquire, never nested.
    """
    clean = (query or "").strip()
    if not clean:
        return clean
    slot = resolve_slot(role)
    try:
        prompt_text = PromptManager.render(
            "rewrite_query.j2",
            query=clean,
            history_block=(history_text or "No prior conversation."),
            filenames_block=(filenames_block or "No attached files."),
        )
    except Exception as e:
        logger.warning("rewrite_skipped_prompt err=%s", e)
        return clean

    if slot == "worker":
        try:
            out = _run_rewrite_worker(prompt_text, max_tokens)
        except RuntimeError as e:
            if "Busy" in str(e):
                logger.warning("rewrite_skipped_busy slot=%s err=%s", slot, e)
            else:
                logger.warning("rewrite_skipped_error slot=%s err=%s", slot, e)
            return clean
        except Exception as e:
            logger.warning("rewrite_skipped_error slot=%s err=%s", slot, e)
            return clean
        if not (out or "").strip():
            logger.warning("rewrite_skipped_empty slot=%s", slot)
            return clean
        return out.strip()

    if LlamaEngine.get_instance("chat").is_generating():
        logger.warning("rewrite_skipped_busy slot=chat")
        return clean
    budget_s = timeout if timeout is not None else config.SUMMARY_TIMEOUT_S
    future = _REWRITE_POOL.submit(_run_rewrite_chat, prompt_text, max_tokens)
    try:
        out = future.result(timeout=budget_s)
    except TimeoutError:
        logger.warning("rewrite_timeout slot=%s budget_s=%s", slot, budget_s)
        future.add_done_callback(lambda _f: _f.exception())
        return clean
    except RuntimeError as e:
        if "Busy" in str(e):
            logger.warning("rewrite_skipped_busy slot=%s err=%s", slot, e)
        else:
            logger.warning("rewrite_skipped_error slot=%s err=%s", slot, e)
        return clean
    except Exception as e:
        logger.warning("rewrite_skipped_error slot=%s err=%s", slot, e)
        return clean
    if not (out or "").strip():
        logger.warning("rewrite_skipped_empty slot=%s", slot)
        return clean
    return out.strip()


def _memory_shares(
    sem_hit: bool, epi_hit: bool, cap: int
) -> Tuple[int, int, bool]:
    """(sem_cap, epi_cap, epi_first) split of mem_cap for gate evidence.

    Epi-only hit fits episodic first up to 70 pct, sem-only hit fits
    semantic first up to 70 pct; both/neither keeps semantic-first on
    the full cap. M5 tag buffer reuses this for its own carve.
    """
    cap = max(0, int(cap))
    dominant = int(cap * MEMORY_DOMINANT_SHARE)
    if epi_hit and not sem_hit:
        return (max(0, cap - dominant), dominant, True)
    if sem_hit and not epi_hit:
        return (dominant, max(0, cap - dominant), False)
    return (cap, cap, False)


def _combine_memory(
    sem_text: str,
    ep_texts: Optional[List[str]],
    sem_hit: bool = False,
    epi_hit: bool = False,
    cap: Optional[int] = None,
) -> str:
    """One memory block inside mem_cap, split per query evidence.

    Single-side input takes the full cap, as today. With both sides
    the dominant evidence fits first (up to 70 pct) and the other
    takes the remainder; both/neither keeps semantic-first.
    """
    limit = config.EFFECTIVE_MEMORY_TOKENS if cap is None else max(0, int(cap))
    sem_clean = (sem_text or "").strip()
    ep_lines = [
        f"earlier: {(s or '').strip()}" for s in ep_texts or [] if (s or "").strip()
    ]
    ep_joined = "\n".join(ep_lines).strip()
    if sem_clean and not ep_joined:
        return _fit_memory(sem_clean, limit)[0].strip()
    if ep_joined and not sem_clean:
        return _fit_memory(ep_joined, limit)[0].strip()
    if not sem_clean and not ep_joined:
        return ""
    sem_cap, epi_cap, epi_first = _memory_shares(sem_hit, epi_hit, limit)
    if epi_first:
        ep_block, epi_used = _fit_memory(ep_joined, epi_cap)
        sem_block, sem_used = _fit_memory(sem_clean, max(0, limit - epi_used))
    else:
        sem_block, sem_used = _fit_memory(sem_clean, sem_cap)
        ep_block, epi_used = _fit_memory(ep_joined, max(0, limit - sem_used))
    logger.debug(
        "memory split sem_hit=%s epi_hit=%s cap=%s sem_used=%s epi_used=%s",
        sem_hit,
        epi_hit,
        limit,
        sem_used,
        epi_used,
    )
    return "\n".join(p for p in (sem_block, ep_block) if p).strip()


class RagState(TypedDict, total=False):
    query: str
    history: List[Dict[str, str]]
    conversation_id: Optional[uuid.UUID]
    decision: RouteDecision
    hits: List[SearchHit]
    messages: List[Dict[str, str]]


class RagGraph:
    """Entry decider, then retrieve and build around one RagService."""

    def __init__(
        self,
        db: Session,
        rag: Optional[RagService] = None,
        decider: Optional[Decider] = None,
        llm: Optional[LLMService] = None,
        top_k: Optional[int] = None,
    ):
        # Shared embed slot per graph: unloaded after the run so only one
        # model is resident on the 8GB box. Injected fakes skip this.
        self._owns_engine = rag is None
        engine = EmbeddingEngine.get_instance("embed") if self._owns_engine else None
        self.rag = rag or RagService(db, engine=engine)
        self.decider = decider or Decider(db)
        self.llm = llm or LLMService()
        self.top_k = top_k if top_k is not None else top_k_for_ctx(config.EFFECTIVE_N_CTX)

        graph = StateGraph(RagState)
        graph.add_node("decide", self._decide)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build", self._build)
        graph.set_entry_point("decide")
        graph.add_conditional_edges(
            "decide", self._route, {"rag": "retrieve", "direct": "build"}
        )
        graph.add_edge("retrieve", "build")
        graph.add_edge("build", END)
        self._app = graph.compile()

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> Dict[str, Any]:
        """Run decider then retrieval/build; fail-open DIRECT on any error."""
        logger.debug("Graph run started - query='%.100s'", query)
        try:
            try:
                out = self._app.invoke(
                    {
                        "query": query,
                        "history": history or [],
                        "conversation_id": conversation_id,
                    }
                )
            finally:
                if self._owns_engine:
                    self.rag.engine.unload()
        except Exception as e:
            logger.warning("Graph failed - falling back to DIRECT: %s", e)
            return {
                "decision": RouteDecision(route="DIRECT", reason="graph fallback"),
                "hits": [],
                "messages": self.llm.build_chat_messages(history or [], query),
                "route": "DIRECT",
            }
        decision = out.get("decision") or RouteDecision(route="DIRECT", reason="empty")
        out["decision"] = decision
        out["route"] = decision.route
        logger.info(
            "RAG decision - route=%s, hits=%s, reason='%s'",
            decision.route,
            len(out.get("hits", [])),
            decision.reason,
        )
        return out

    def _decide(self, state: RagState) -> Dict[str, Any]:
        try:
            history = state.get("history") or []
            decision = self.decider.decide(
                state.get("query", ""),
                state.get("conversation_id"),
                history=history[-2:],
            )
        except Exception as e:
            logger.warning("Decider failed - falling back to DIRECT: %s", e)
            decision = RouteDecision(route="DIRECT", reason="decider error")

        return {"decision": decision}

    def _route(self, state: RagState) -> str:
        decision = state.get("decision")
        if decision is not None and decision.route == "RAG":
            return "rag"
        return "direct"

    def _attached_filenames_block(
        self, conversation_id: Optional[uuid.UUID] = None
    ) -> str:
        """Attached filenames for the rewrite prompt, fail-open to none."""
        try:
            if conversation_id is not None:
                rows = self.rag.docs.list_by_conversation(conversation_id, limit=8)
            else:
                rows = self.rag.docs.list_recent(limit=8)
        except Exception as e:
            logger.debug("rewrite filenames skipped: %s", e)
            return "No attached files."
        try:
            names = [
                (getattr(d, "filename", "") or "").strip() for d in rows or []
            ]
        except Exception as e:
            logger.debug("rewrite filenames skipped: %s", e)
            return "No attached files."
        names = [n for n in names if n][:8]
        if not names:
            return "No attached files."
        return "\n".join(f"- {n}" for n in names)

    def _maybe_rewrite_query(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        conversation_id: Optional[uuid.UUID] = None,
    ) -> str:
        """Laya rewrite_needed gate; returns retrieval text, never raises.

        Empty history skips the gate (no Laya, no SLM). Below threshold
        returns the original with zero SLM cost. At or above threshold the
        worker/chat SLM expands the follow-up; any failure returns the
        original. Retrieval-internal only: the caller keeps the user turn.
        """
        clean = (query or "").strip()
        if not clean:
            return clean
        recent = list(history or [])[-2:]
        if not recent:
            logger.debug("rewrite=false reason=empty-history")
            return clean
        history_text = _rewrite_history_text(recent)
        try:
            res = LayaService.get_instance().predict(
                {"query": clean, "history": history_text},
                {"rewrite_needed": REWRITE_QUESTION},
            )
            answers = (res or {}).get("answers") or {}
            ans = answers.get("rewrite_needed") or {}
            try:
                p_true = float(ans.get("noul", 0.0))
            except (TypeError, ValueError):
                p_true = 0.0
        except Exception as e:
            logger.debug("rewrite_skipped_laya err=%s", e)
            return clean
        try:
            threshold = float(config.LAYA_REWRITE_THRESHOLD)
        except Exception:
            threshold = 0.8
        if p_true < threshold:
            logger.debug(
                "rewrite=false p_true=%.2f threshold=%.2f", p_true, threshold
            )
            return clean
        try:
            filenames_block = self._attached_filenames_block(conversation_id)
        except Exception:
            filenames_block = "No attached files."
        try:
            rewritten = _rewrite_via_slm(
                clean, history_text, filenames_block, role=None
            )
        except Exception as e:
            logger.debug("rewrite_skipped_slm err=%s", e)
            return clean
        final = (rewritten or "").strip() or clean
        try:
            orig_toks = count_tokens(clean)
            new_toks = count_tokens(final)
            logger.debug(
                "rewrite=true p_true=%.2f orig_toks=%s new_toks=%s delta=%s",
                p_true,
                orig_toks,
                new_toks,
                new_toks - orig_toks,
            )
        except Exception:
            logger.debug("rewrite=true p_true=%.2f", p_true)
        return final

    def _retrieve(self, state: RagState) -> Dict[str, Any]:
        # Rewrite runs here before any generation acquire (same rule as
        # summaries: never nested in acquire). Final text feeds search and
        # grading; state query stays original for the user-facing build.
        try:
            raw = state.get("query", "")
            history = state.get("history") or []
            conversation_id = state.get("conversation_id")
            try:
                final = self._maybe_rewrite_query(raw, history, conversation_id)
            except Exception as e:
                logger.debug("rewrite_skipped err=%s", e)
                final = (raw or "").strip()
            hits = self.rag.search(
                final,
                top_k=self.top_k,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.warning("Retrieval failed - falling back to DIRECT: %s", e)
            return {
                "hits": [],
                "decision": RouteDecision(route="DIRECT", reason="retrieval error"),
            }

        decision = state.get("decision")

        if not hits and decision is not None and decision.route == "RAG":
            decision = RouteDecision(
                route="DIRECT", reason="no hits - ask with filename"
            )

        return {"hits": hits, "decision": decision}

    def _memory_text(
        self, query: str, conversation_id: Optional[uuid.UUID] = None
    ) -> Optional[str]:
        """Semantic facts plus episodic summaries in one block, None when none.

        Cheap overlap pools first; the embed engine stays untouched on a
        memory-less turn. Otherwise one batched embed (query plus capped
        candidates) feeds vector-blended recall; any embed failure falls
        back to overlap-only.
        """
        from services.episodic_service import EpisodicService

        db = self.rag.db
        try:
            sem_overlap = SemanticMemoryRepository(db).find_relevant(query, limit=5)
        except Exception as e:
            logger.debug("memory recall skipped: %s", e)
            sem_overlap = []
        try:
            sem_rows = list(SemanticMemoryRepository(db).list_all(limit=100) or [])
        except Exception as e:
            logger.debug("semantic pool skipped: %s", e)
            sem_rows = []
        try:
            if conversation_id is not None:
                epi_rows = list(
                    EpisodicService(db).repo.list_recent_for_query(
                        conversation_id, limit=10
                    )
                    or []
                )
            else:
                epi_rows = []
        except Exception as e:
            logger.debug("episodic pool skipped: %s", e)
            epi_rows = []
        evidence = memory_evidence(query, memories=sem_overlap, epi_rows=epi_rows)
        if not evidence.needs_memory and not epi_rows and not sem_rows:
            return None  # memory-less turn, engine untouched
        sem_items = []
        for row in sem_rows:
            key = getattr(row, "key", "") or ""
            text = memory_embed_text(f"{key}: {getattr(row, 'value', '') or ''}")
            if text.strip():
                sem_items.append((key, text))
        epi_items = []
        for row in epi_rows:
            text = memory_embed_text(getattr(row, "summary", "") or "")
            if text.strip():
                epi_items.append((str(getattr(row, "id", "")), text))
        vecs = None
        try:
            texts = (
                [query or ""]
                + [text for _, text in sem_items]
                + [text for _, text in epi_items]
            )
            vecs = self.rag.engine.embed(texts)
            if (
                not isinstance(vecs, list)
                or len(vecs) != len(texts)
                or not all(isinstance(v, list) and v for v in vecs)
            ):
                raise ValueError("embed shape mismatch")
        except Exception as e:
            logger.debug("memory embed skipped, overlap-only: %s", e)
            vecs = None
        if vecs is None:
            rows = sem_overlap
            ep_texts: List[str] = []
            if conversation_id is not None:
                try:
                    ep_texts = EpisodicService(db).recall(
                        conversation_id, query, limit=3
                    )
                except Exception as e:
                    logger.debug("episodic recall skipped, semantic-only: %s", e)
                    ep_texts = []
        else:
            query_vector = vecs[0]
            sem_vecs = {
                key: vec for (key, _), vec in zip(sem_items, vecs[1:])
            }
            epi_vecs = {
                key: vec
                for (key, _), vec in zip(epi_items, vecs[1 + len(sem_items):])
            }
            try:
                rows = SemanticMemoryRepository(db).find_relevant(
                    query,
                    limit=5,
                    query_vector=query_vector,
                    candidate_vectors=sem_vecs,
                )
            except Exception as e:
                logger.debug("memory recall skipped: %s", e)
                rows = sem_overlap
            ep_texts = []
            if conversation_id is not None:
                try:
                    ep_texts = EpisodicService(db).recall(
                        conversation_id,
                        query,
                        limit=3,
                        query_vector=query_vector,
                        candidate_vectors=epi_vecs,
                    )
                except Exception as e:
                    logger.debug("episodic recall skipped, semantic-only: %s", e)
                    ep_texts = []
        lines = [
            f"{r.key}: {r.value}"
            for r in rows or []
            if (getattr(r, "value", "") or "").strip()
        ]
        sem_text = "\n".join(lines).strip()
        text = _combine_memory(
            sem_text, ep_texts, sem_hit=evidence.sem_hit, epi_hit=evidence.epi_hit
        )
        return text or None

    def _project_summary_line(
        self, conversation_id: Optional[uuid.UUID] = None
    ) -> List[str]:
        """Project shared summary as one labeled line, [] when unlinked.

        Rides the history remainder with sibling lines, never the memory
        carve, so own-chat episodic keeps priority. Fail-open to [].
        """
        if conversation_id is None:
            return []
        try:
            from repository.chat_repository import ChatRepository
            from repository.project_repository import ProjectRepository

            db = self.rag.db
            current = ChatRepository(db).get_by_id(conversation_id)
            project_id = getattr(current, "project_id", None) if current else None
            if project_id is None:
                return []
            proj = ProjectRepository(db).get_by_id(project_id)
            summary = ((getattr(proj, "summary", "") or "").strip()) if proj else ""
            if not summary:
                return []
            name = ((getattr(proj, "name", "") or "").strip()) or "project"
            return [f"Project {name} summary: {summary}"][:PROJECT_SUMMARY_MAX_LINES]
        except Exception as e:
            logger.debug("project summary recall skipped: %s", e)
            return []

    def _tag_sibling_lines(
        self, conversation_id: Optional[uuid.UUID] = None
    ) -> List[str]:
        """Latest episodic summaries from same-tag sibling chats.

        One labeled line per sibling, newest siblings first, capped at
        TOPIC_SIBLING_MAX_LINES total. Untagged chats return no lines so
        their prompts stay byte-identical. Any error fails open to [].
        """
        if conversation_id is None:
            return []
        try:
            from repository.chat_repository import ChatRepository
            from repository.episodic_repository import EpisodicMemoryRepository

            db = self.rag.db
            current = ChatRepository(db).get_by_id(conversation_id)
            tag = ((getattr(current, "tag", None) or "").strip()) if current else ""
            if not tag:
                return []
            siblings = ChatRepository(db).list_by_tag(
                tag, limit=10, exclude_id=conversation_id
            )
            epi_repo = EpisodicMemoryRepository(db)
            lines: List[str] = []
            for sib in siblings or []:
                if len(lines) >= TOPIC_SIBLING_MAX_LINES:
                    break
                try:
                    rows = epi_repo.list_recent_for_query(sib.id, limit=1)
                except Exception as e:
                    logger.debug("sibling episodic skipped conv=%s: %s", sib.id, e)
                    continue
                if not rows:
                    continue
                summary = (getattr(rows[0], "summary", "") or "").strip()
                if summary:
                    lines.append(f"Earlier in project {tag}: {summary}")
            return lines[:TOPIC_SIBLING_MAX_LINES]
        except Exception as e:
            logger.debug("tag sibling recall skipped: %s", e)
            return []

    def _build(self, state: RagState) -> Dict[str, List[Dict[str, str]]]:
        """Grounded or plain messages; memory keeps its carve first.

        Effective priority order: memory carve (sem/epi split per gate
        evidence in _combine_memory) first, then RAG hits, then history.
        Sibling tag lines ride inside the history cap only: own history
        is fitted first and siblings fill the remainder (see _fit_topic),
        so own-chat episodic in the memory carve always keeps priority.
        Route RAG with hits builds grounded messages, otherwise plain
        chat. allocate() signature unchanged - needs_memory follows the
        memory block, so strong memory evidence never loses its carve
        to RAG hits.
        """
        query = state.get("query", "")
        history = state.get("history", [])
        decision = state.get("decision")
        hits = state.get("hits", [])
        memory_text = self._memory_text(query, state.get("conversation_id"))
        topic_lines = self._project_summary_line(
            state.get("conversation_id")
        ) + (self._tag_sibling_lines(state.get("conversation_id")) or [])

        if decision is not None and decision.route == "RAG" and hits:
            messages = self.rag.build_messages(
                query, hits, history, memory_text=memory_text,
                topic_lines=topic_lines or None,
            )
        else:
            messages = self.rag.build_messages(
                query, None, history, memory_text=memory_text,
                topic_lines=topic_lines or None,
            )
        return {"messages": messages}
