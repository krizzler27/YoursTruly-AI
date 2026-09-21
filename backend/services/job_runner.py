"""Generic background FIFO: submit fast, one daemon worker drains serially.

Child jobs own their payload and processing. This module only moves
opaque jobs from submit to a handler, one at a time, with per-job
error isolation and a worker watchdog that restarts a dead thread.
"""

import threading
from queue import Queue
from typing import Callable

from core.logging import get_logger

logger = get_logger(__name__)


class JobRunner:
    """Single FIFO worker over opaque jobs. Subclass or pass a handler."""

    def __init__(self, handler: Callable[[object], None], name: str = "jobs"):
        self._handler = handler
        self._name = name
        self._jobs: Queue = Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, job: object) -> None:
        """Enqueue without blocking. Restarts a dead worker first."""
        self.ensure_worker()
        self._jobs.put(job)

    def queue_depth(self) -> int:
        """Queued jobs behind the running one. Never negative."""
        return max(0, self._jobs.qsize() - 1)

    @property
    def queue(self) -> Queue:
        """Underlying queue, exposed for depth checks and tests."""
        return self._jobs

    def ensure_worker(self) -> None:
        """Start the worker once. Restart it when dead."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker_loop, name=f"jobrunner-{self._name}", daemon=True
            )
            self._thread.start()

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                self._handler(job)
            except Exception as e:
                logger.warning("worker job failed: %s", e, exc_info=True)
            finally:
                self._jobs.task_done()
