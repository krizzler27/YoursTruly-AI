"""Background episodic rollup: submit fast, summarize off the request path.

Flow: chat_api saves the assistant reply then RollupJob.submit enqueues one
(conversation_id, request_id) tuple and returns in microseconds - no model
touch, no engine acquire. A single daemon worker (own BackgroundJobs _runner)
takes jobs FIFO, opens its own SessionLocal, reloads history, and calls
EpisodicService.rollup_if_needed. Everything fail-open: a missed rollup only
means the next turn truncates oldest instead.
"""

from queue import Queue
from uuid import UUID

from sqlalchemy.orm import Session

from db.db_engine import SessionLocal  # worker thread boundary: own session
from core.logging import get_logger, get_request_id, set_conversation_id, set_request_id
from services.background_jobs import BackgroundJobs

logger = get_logger(__name__)


class RollupJob:
    """Summarize older turns serially off the request path. run() is the worker entry."""

    def submit(self, db: Session, conversation_id: UUID) -> None:
        """Enqueue one rollup; never touches the model, returns fast."""
        try:
            from repository.chat_repository import ChatRepository

            if ChatRepository(db).get_by_id(conversation_id) is None:
                logger.debug("rollup skipped, unknown conversation=%s", conversation_id)
                return
            _runner.submit((str(conversation_id), get_request_id()))
        except Exception as e:
            logger.debug("rollup submit skipped: %s", e)

    def run(self, job: tuple) -> None:
        """Roll up one conversation: reload history, summarize prefix, store."""
        conversation_id, request_id = job
        set_request_id(request_id)
        set_conversation_id(conversation_id)
        db = SessionLocal()
        try:
            from core.context_budget import allocate
            from services.chat_services import ChatServices
            from services.episodic_service import EpisodicService

            conv_id = UUID(str(conversation_id))
            history = ChatServices(db).get_history(conv_id, limit=100)
            if len(history) <= 2:
                return
            usable = allocate(route="DIRECT", needs_memory=False, query_tokens=0)["usable"]
            EpisodicService(db).rollup_if_needed(conv_id, history, usable)
        except Exception as e:
            logger.debug("episodic rollup skipped: %s", e)
        finally:
            db.close()


rollup_job = RollupJob()
_jobs: Queue = Queue()
_runner = BackgroundJobs(handler=rollup_job.run, name="rollup", queue=_jobs)
