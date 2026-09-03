"""Download manager — non-blocking single GGUF download with polling.

Runs `llmfit download` in a daemon Thread so POST /api/models/download
returns 202 instantly. Frontend polls GET /api/downloads/{id}.
Only one download at a time — second request gets 409.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
SAVED_RE = re.compile(r"Model saved to:\s*([^\s]+)")


class DownloadManager:
    """Thread-safe single-job tracker."""

    _instance: Optional[DownloadManager] = None
    _lock = threading.Lock()

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        self._procs: Dict[str, subprocess.Popen] = {}

    @classmethod
    def get_instance(cls) -> DownloadManager:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _is_already_installed(self, repo_id: str, quant: Optional[str]) -> bool:
        """Check models_dir for existing GGUF matching repo+quant (heuristic)."""
        try:
            models_dir = Path(config.LLAMA_MODEL_PATH)
            if not models_dir.exists():
                return False
            repo_base = repo_id.strip().split("/")[-1].lower().replace("-gguf", "").replace(".gguf", "")
            # core token e.g. qwen2.5-3b from qwen2.5-3b-instruct
            core = "-".join(repo_base.split("-")[:2]) if "-" in repo_base else repo_base[:8]
            quant_norm = (quant or "").lower().replace("_", "-")
            for p in models_dir.glob("*.gguf"):
                n = p.name.lower()
                if core and core not in n:
                    continue
                if quant_norm and quant_norm.replace("-", "") not in n.replace("-", "").replace("_", ""):
                    # allow q4_k_m to match q4km etc
                    if quant_norm not in n and quant_norm.replace("-", "_") not in n:
                        continue
                return True
            return False
        except Exception:
            return False

    def start(self, repo_id: str, quant: Optional[str] = None) -> Dict[str, Any]:
        """Enqueue single download; raises RuntimeError 409 if busy or already installed."""
        if not repo_id or not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
        repo_id = repo_id.strip()
        if quant:
            quant = quant.strip().upper() or None

        with self._jobs_lock:
            if any(j["status"] in ("queued", "downloading") for j in self._jobs.values()):
                raise RuntimeError("A download is already in progress — please wait")
        if self._is_already_installed(repo_id, quant):
            raise RuntimeError("Model already installed")

        job_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        job: Dict[str, Any] = {
            "id": job_id,
            "repo_id": repo_id,
            "quant": quant,
            "status": "queued",
            "progress": 0,
            "path": None,
            "filename": None,
            "error": None,
            "logs": "",
            "created_at": now,
            "updated_at": now,
            "bytes_downloaded": 0,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job

        t = threading.Thread(target=self._run, args=(job_id, repo_id, quant), daemon=True)
        t.start()
        return dict(job)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._jobs_lock:
            j = self._jobs.get(job_id)
            return dict(j) if j else None

    def list_all(self) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            return sorted([dict(v) for v in self._jobs.values()], key=lambda x: x["created_at"], reverse=True)

    def cancel(self, job_id: str) -> bool:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in ("completed", "failed"):
                return False
        proc = self._procs.get(job_id)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        with self._jobs_lock:
            if job_id in self._jobs and self._jobs[job_id]["status"] in ("queued", "downloading"):
                self._jobs[job_id]["status"] = "failed"
                self._jobs[job_id]["error"] = "Cancelled by user"
                self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        return True

    def _update(self, job_id: str, **kw: Any) -> None:
        with self._jobs_lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(kw)
                self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _ensure_runner() -> List[str]:
        if shutil.which("llmfit"):
            return ["llmfit"]
        if shutil.which("uvx"):
            return ["uvx", "llmfit"]
        raise FileNotFoundError("llmfit not found: scoop install llmfit / brew install llmfit")

    @staticmethod
    def _build_cmd(runner: List[str], repo_id: str, quant: Optional[str], output_dir: str) -> List[str]:
        if not repo_id or not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
        repo_id = repo_id.strip()
        if quant:
            quant = quant.strip().upper() or None
        cmd = runner + ["download", repo_id]
        if quant:
            cmd += ["--quant", quant]
        cmd += ["--output-dir", output_dir]
        return cmd

    def _run(self, job_id: str, repo_id: str, quant: Optional[str]) -> None:
        models_dir = Path(config.LLAMA_MODEL_PATH)
        models_dir.mkdir(parents=True, exist_ok=True)
        self._update(job_id, status="downloading", progress=2)

        try:
            runner = self._ensure_runner()
            cmd = self._build_cmd(runner, repo_id, quant, str(models_dir))
        except (FileNotFoundError, ValueError) as e:
            self._update(job_id, status="failed", error=str(e), progress=0)
            return

        logs: List[str] = []
        last_pct = 2
        proc: Optional[subprocess.Popen] = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
            self._procs[job_id] = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                logs.append(line)
                m = PCT_RE.search(line)
                if m:
                    try:
                        pct = max(0, min(95, float(m.group(1))))
                        if pct > last_pct:
                            last_pct = pct
                            self._update(job_id, progress=int(pct), logs="".join(logs)[-3000:])
                    except ValueError:
                        pass
                if len(logs) % 20 == 0:
                    self._update(job_id, logs="".join(logs)[-3000:])

            proc.wait(timeout=600)
            out = "".join(logs)
            if proc.returncode != 0:
                err = out[-2000:] or f"llmfit download failed for {repo_id} (exit {proc.returncode})"
                self._update(job_id, status="failed", error=err, logs=out[-3000:], progress=0)
                return

            m_saved = SAVED_RE.search(out)
            saved_path: Optional[Path] = Path(m_saved.group(1).strip()) if m_saved else None
            if not saved_path or not saved_path.exists():
                cands = [p for p in models_dir.glob("*.gguf") if p.is_file()]
                saved_path = max(cands, key=lambda p: p.stat().st_mtime) if cands else None
            if saved_path and saved_path.exists():
                self._update(
                    job_id,
                    status="completed",
                    progress=100,
                    path=str(saved_path),
                    filename=saved_path.name,
                    logs=out[-3000:],
                    bytes_downloaded=saved_path.stat().st_size,
                )
            else:
                self._update(job_id, status="failed", error=out[-2000:] or "Download finished but no GGUF found", logs=out[-3000:], progress=0)

        except subprocess.TimeoutExpired:
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
            self._update(job_id, status="failed", error=f"Download timeout for {repo_id} (600s)", logs="".join(logs)[-3000:])
        except Exception as e:
            self._update(job_id, status="failed", error=str(e), logs="".join(logs)[-3000:] if logs else str(e), progress=0)
        finally:
            self._procs.pop(job_id, None)
