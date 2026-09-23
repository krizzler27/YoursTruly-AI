"""A queue drained by one background thread, serially.

Submit returns fast while work happens off the request path.
The worker invokes one callable per queued item.
"""

import threading
from queue import Queue
from typing import Callable

from core.logging import get_logger

logger = get_logger(__name__)


class BackgroundJobs:
    """One background thread invoking a callable per queued item."""

    def __init__(self, handler: Callable[[object], None], name: str = "jobs", queue: Queue | None = None):
        self._handler = handler
        self._name = name
        self._jobs: Queue = queue or Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, job: object) -> None:
        """Enqueue without blocking. Restarts a dead worker first."""
        self.ensure_worker()
        self._jobs.put(job)

    def ensure_worker(self) -> None:
        """Start the worker once. Restart it when dead."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=self._worker_loop, name=f"background-{self._name}", daemon=True
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
