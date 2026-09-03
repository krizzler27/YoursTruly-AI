from typing import Any, Dict, List, Optional, Set
import subprocess
import shutil
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

TRUSTED_PROVIDERS: Set[str] = {
    "meta",
    "google",
    "alibaba",
    "deepseek",
    "mistral",
    "microsoft",
}


class LLMFitServices:
    """Wraps llmfit CLI for hardware-aware model discovery and local GGUF I/O."""

    def __init__(
        self,
        trusted_providers: Optional[Set[str]] = None,
    ):
        """Init with allowed providers; defaults to TRUSTED_PROVIDERS."""
        self.trusted_providers = {p.lower() for p in (trusted_providers or TRUSTED_PROVIDERS)}

    def _ensure_runner(self) -> List[str]:
        """Return CLI prefix: `llmfit` if installed else `uvx llmfit`; raises if missing."""
        if shutil.which("llmfit"):
            return ["llmfit"]
        if shutil.which("uvx"):
            return ["uvx", "llmfit"]
        raise FileNotFoundError("llmfit not found: scoop install llmfit / brew install llmfit")

    def system_info(self) -> Dict[str, Any]:
        """Run `llmfit system --json` and return `{system, raw}` for ledger UI."""
        runner = self._ensure_runner()
        res = subprocess.run(
            runner + ["system", "--json"],
            capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=15,
        )
        data = json.loads(res.stdout)
        return {"system": data.get("system", data), "raw": data}

    def discover_and_classify(
        self,
        limit: int = 20,
        include_community: bool = False,
        perfect_only: bool = False,
        providers: Optional[List[str]] = None,
        sort: str = "score",
    ) -> Dict[str, Any]:
        """Run `llmfit fit --json` with filters, return `{system, meta, catalog, models}` ranked for this hardware."""
        from schemas.api_schemas import SORT_ALIASES

        sort = SORT_ALIASES.get(sort.strip().lower(), sort.strip().lower()) if sort else "score"
        if providers is not None and len(providers) == 0:
            providers = None  # [] -> trusted
        runner = self._ensure_runner()
        cmd = runner + ["fit", "--json", "--limit", str(limit), "--sort", sort]
        if perfect_only:
            cmd += ["--perfect"]
        if providers is not None:
            if providers:
                cmd += ["--providers", ",".join(providers)]
        elif not include_community:
            cmd += ["--providers", ",".join(sorted(self.trusted_providers))]

        res = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        raw = json.loads(res.stdout)
        models = raw.get("models", raw) if isinstance(raw, dict) else raw
        system = raw.get("system") if isinstance(raw, dict) else None
        catalog, flat = self._classify(models)
        return {
            "system": system,
            "meta": {"cmd": " ".join(cmd), "limit": limit, "include_community": include_community, "perfect_only": perfect_only, "total": len(models)},
            "catalog": catalog,
            "models": flat,
        }

    def recommend(
        self,
        limit: int = 20,
        use_case: Optional[str] = None,
        min_fit: str = "marginal",
        runtime: str = "any",
        providers: Optional[List[str]] = None,
        sort: str = "score",
        include_community: bool = False,
        perfect_only: bool = False,
    ) -> Dict[str, Any]:
        """Run `llmfit recommend --json` filtered by use_case (general/coding/...), same ranking as fit."""
        from schemas.api_schemas import SORT_ALIASES

        sort = SORT_ALIASES.get(sort.strip().lower(), sort.strip().lower()) if sort else "score"
        if providers is not None and len(providers) == 0:
            providers = None
        runner = self._ensure_runner()
        cmd = runner + ["recommend", "--json", "--limit", str(limit), "--sort", sort]
        if use_case:
            cmd += ["--use-case", use_case]
        if min_fit:
            cmd += ["--min-fit", min_fit]
        if runtime and runtime != "any":
            cmd += ["--runtime", runtime]
        if perfect_only:
            cmd += ["--perfect"]
        # recommend respects same provider filtering as fit
        if providers is not None:
            if providers:
                cmd += ["--providers", ",".join(providers)]
        elif not include_community:
            cmd += ["--providers", ",".join(sorted(self.trusted_providers))]

        res = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        raw = json.loads(res.stdout)
        models = raw.get("models", raw) if isinstance(raw, dict) else raw
        system = raw.get("system") if isinstance(raw, dict) else None
        catalog, flat = self._classify(models)
        return {
            "system": system,
            "meta": {"cmd": " ".join(cmd), "limit": limit, "use_case": use_case, "min_fit": min_fit, "total": len(models)},
            "catalog": catalog,
            "models": flat,
        }

    def _classify(self, models: List[dict]) -> tuple[Dict[str, List[dict]], List[dict]]:
        """Normalize llmfit raw models to UI shape and bucket by fit_level."""
        catalog: Dict[str, List[dict]] = {"perfect": [], "runnable": [], "others": []}
        flat: List[dict] = []
        for m in models:
            gguf_sources = m.get("gguf_sources") or []
            item = {
                "name": m.get("name"),
                "provider": m.get("provider"),
                "fit_level": m.get("fit_level"),
                "quant": m.get("best_quant"),
                "run_mode": m.get("run_mode"),
                "ollama_name": m.get("ollama_name"),
                "gguf_sources": gguf_sources,
                "hf_repo": (gguf_sources[0].get("repo") if gguf_sources else None) or m.get("hf_repo") or m.get("repo_id"),
                "hf_file": m.get("hf_file") or m.get("filename"),
                "best_quant": m.get("best_quant"),
                "disk_size_gb": m.get("disk_size_gb"),
                "context_length": m.get("effective_context_length") or m.get("context_length"),
                "vram_gb": m.get("memory_required_gb"),
                "tps": m.get("estimated_tps"),
                "score": m.get("score"),
                "use_case": m.get("use_case"),
                "category": m.get("category"),
            }
            flat.append(item)
            lvl = str(m.get("fit_level", "")).lower()
            if lvl == "perfect":
                catalog["perfect"].append(item)
            elif lvl in ("good", "marginal"):
                catalog["runnable"].append(item)
            else:
                catalog["others"].append(item)
        return catalog, flat

    def list_local(self) -> List[str]:
        """Scan `~/.yourstrulyai/models` for `*.gguf`; returns sorted filenames for UI picker."""
        from config import config
        d = Path(config.LLAMA_MODEL_PATH)
        try:
            return sorted([p.name for p in d.glob("*.gguf") if p.is_file()])
        except Exception as e:
            logger.warning("list_local failed %s: %s", d, e)
            return []

    def download(self, repo_id: str, filename: str = "", quant: Optional[str] = None) -> Dict[str, Any]:
        """Download GGUF via `llmfit download <repo> [--quant] --output-dir <models>`; dedups, renames, returns `{status, path}`."""
        if not repo_id or not repo_id.strip():
            raise ValueError("repo_id cannot be empty")
        repo_id = repo_id.strip()
        filename = (filename or "").strip()
        if filename and ("/" in filename or "\\" in filename or ".." in filename):
            raise ValueError("Invalid filename")
        if filename and not filename.lower().endswith(".gguf"):
            raise ValueError("filename must be .gguf")

        # derive quant from filename if not explicit
        if not quant and filename:
            # e.g. Llama-3.2-3B-Q4_K_M.gguf → Q4_K_M
            import re
            m = re.search(r"(Q\d[_\w]*|IQ\d_\w+|BF16|F16|F32)", filename, re.IGNORECASE)
            if m:
                quant = m.group(1).upper()

        from config import config
        d = Path(config.LLAMA_MODEL_PATH)
        # if filename provided and already exists, skip
        if filename:
            dest = d / filename
            if dest.exists() and dest.stat().st_size > 1024 * 1024:
                return {"status": "exists", "path": str(dest), "repo_id": repo_id, "filename": filename, "quant": quant}

        runner = self._ensure_runner()
        cmd = runner + ["download", repo_id]
        if quant:
            cmd += ["--quant", quant]
        cmd += ["--output-dir", str(d)]
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=600
            )
            if res.returncode != 0:
                raise RuntimeError(res.stderr or res.stdout or f"llmfit download failed for {repo_id}")
            import re
            m = re.search(r"Model saved to:\s*([^\s]+)", res.stdout)
            saved_path: Optional[Path] = Path(m.group(1).strip()) if m else None
            # llmfit with --output-dir writes directly to d; fallback scan d for newest .gguf
            if not saved_path or not saved_path.exists():
                candidates = sorted(d.glob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
                saved_path = candidates[0] if candidates else None
            if not saved_path or not saved_path.exists():
                return {"status": "ok", "path": str(d), "repo_id": repo_id, "filename": filename or "", "quant": quant, "output": res.stdout[-1500:]}
            # honor explicit filename — rename if llmfit chose different name
            if filename and saved_path.name != filename:
                dest = d / filename
                try:
                    if dest.exists():
                        dest.unlink()
                    saved_path.rename(dest)
                    saved_path = dest
                except Exception as e:
                    raise RuntimeError(f"Rename to {filename} failed: {e}") from e
            return {"status": "ok", "path": str(saved_path), "repo_id": repo_id, "filename": saved_path.name, "quant": quant, "output": res.stdout[-1500:]}
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Download timeout for {repo_id}: {e}") from e
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Download failed: {e}") from e

    def list_remote_ggufs(self, model: str) -> Dict[str, Any]:
        """Run `llmfit download <model> --list` and return `{model, output, options}` of available quants."""
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        runner = self._ensure_runner()
        cmd = runner + ["download", model.strip(), "--list"]
        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or res.stdout)
        return {"model": model, "output": res.stdout, "options": res.stdout.splitlines()[:20]}

    def delete_local(self, filename: str) -> Dict[str, Any]:
        """Delete `filename` from models dir; blocks path traversal, raises if missing."""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")
        filename = filename.strip()
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError("Invalid filename")
        from config import config
        d = Path(config.LLAMA_MODEL_PATH)
        p = d / filename
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        p.unlink()
        return {"status": "deleted", "filename": filename, "path": str(p)}
