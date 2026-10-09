"""Project shared summary - aggregate member episodic rows in background."""

import uuid
from typing import Optional

from sqlalchemy.orm import Session

from core.logging import get_logger
from repository.chat_repository import ChatRepository
from repository.episodic_repository import EpisodicMemoryRepository
from repository.project_repository import ProjectRepository
import services.summarize_service as summarize_mod

logger = get_logger(__name__)

AGGREGATE_MAX_MEMBERS = 10
AGGREGATE_MAX_ROWS_PER_MEMBER = 3
AGGREGATE_MAX_TOKENS = 128
AGGREGATE_MAX_CHARS = 4000


def aggregate_project(db: Session, conversation_id: uuid.UUID) -> Optional[str]:
    """Refresh one project summary from member episodic rows, None when skipped."""
    try:
        conv = ChatRepository(db).get_by_id(conversation_id)
        project_id = getattr(conv, "project_id", None) if conv else None
        if project_id is None:
            return None
        members = ChatRepository(db).list_by_project(
            project_id, limit=AGGREGATE_MAX_MEMBERS + 1
        )
        ids = [conversation_id] + [
            m.id for m in members or [] if m.id != conversation_id
        ]
        texts = []
        for cid in ids[:AGGREGATE_MAX_MEMBERS]:
            try:
                rows = EpisodicMemoryRepository(db).list_recent_for_query(
                    cid, limit=AGGREGATE_MAX_ROWS_PER_MEMBER
                )
            except Exception as e:
                logger.debug("project member episodic skipped conv=%s: %s", cid, e)
                continue
            for row in rows or []:
                summary = (getattr(row, "summary", "") or "").strip()
                if summary:
                    texts.append(summary)
        if not texts:
            return None
        combined = "\n".join(texts)[:AGGREGATE_MAX_CHARS].strip()
        if not combined:
            return None
        summary = summarize_mod.summarize_text(
            combined, max_tokens=AGGREGATE_MAX_TOKENS, role="worker"
        )
        if not (summary or "").strip():
            return None
        clean = summary.strip()
        ProjectRepository(db).update_summary(project_id, clean)
        logger.info("project aggregate done project=%s", project_id)
        return clean
    except Exception as e:
        logger.debug("project aggregate skipped: %s", e)
        return None
