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
    """Hardware-aware model discovery for local llama.cpp inference.

    Keeps llmfit CLI for system probe + catalog scoring (perfect/runnable),
    but model store is local GGUF in ~/.yourstrulyai/models via huggingface_hub,
    not Ollama daemon.
    """

    def __init__(
        self,
        trusted_providers: Optional[Set[str]] = None,
    ):
        self.trusted_providers = {p.lower() for p in (trusted_providers or TRUSTED_PROVIDERS)}

    def _ensure_runner(self) -> List[str]:
        if shutil.which("llmfit"):
            return ["llmfit"]
        if shutil.which("uvx"):
            return ["uvx", "llmfit"]
        raise FileNotFoundError("llmfit not found: scoop install llmfit / brew install llmfit")

    def _models_dir(self) -> Path:
        # single source from config
        from config import config
        p = Path(config.LLAMA_MODEL_PATH).parent
        p.mkdir(parents=True, exist_ok=True)
        return p

    def system_info(self) -> Dict[str, Any]:
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

        catalog: Dict[str, List[dict]] = {"perfect": [], "runnable": [], "others": []}
        flat: List[dict] = []
        for m in models:
            # llmfit raw uses gguf_sources: [{provider, repo}] and best_quant
            gguf_sources = m.get("gguf_sources") or []
            # keep raw for explore UI to pick repo/filename
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
            }
            flat.append(item)
            lvl = str(m.get("fit_level", "")).lower()
            if lvl == "perfect":
                catalog["perfect"].append(item)
            elif lvl in ("good", "marginal"):
                catalog["runnable"].append(item)
            else:
                catalog["others"].append(item)

        return {
            "system": system,
            "meta": {"cmd": " ".join(cmd), "limit": limit, "include_community": include_community, "perfect_only": perfect_only, "total": len(models)},
            "catalog": catalog,
            "models": flat,
        }

    def list_local(self) -> List[str]:
        """List local GGUFs in ~/.yourstrulyai/models — for UI picker."""
        d = self._models_dir()
        try:
            return sorted([p.name for p in d.glob("*.gguf") if p.is_file()])
        except Exception as e:
            logger.warning("list_local failed %s: %s", d, e)
            return []

    def download(self, repo_id: str, filename: str = "", quant: Optional[str] = None) -> Dict[str, Any]:
        """Download via llmfit CLI to models dir. Replaces `ollama pull` + hf Hub.

        Uses `llmfit download <model> [--quant Q4_K_M]` which auto-selects best quant for hardware.
        repo_id can be HF repo (bartowski/Llama-...-GGUF), known name (llama-3.2-3b), or search query.
        If filename given, quant is derived from filename (e.g. Q4_K_M.gguf → Q4_K_M).
        Downloads to llmfit cache (~/.cache/llama.cpp) then moves to ~/.yourstrulyai/models.
        """
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

        d = self._models_dir()
        # if filename provided and already exists, skip
        if filename:
            dest = d / filename
            if dest.exists() and dest.stat().st_size > 1024 * 1024:
                return {"status": "exists", "path": str(dest), "repo_id": repo_id, "filename": filename, "quant": quant}

        runner = self._ensure_runner()
        cmd = runner + ["download", repo_id]
        if quant:
            cmd += ["--quant", quant]
        # llmfit download respects hardware to pick quant if not specified
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=600
            )
            if res.returncode != 0:
                raise RuntimeError(res.stderr or res.stdout or f"llmfit download failed for {repo_id}")
            # llmfit prints: Model saved to: ~/.cache/llama.cpp/...
            import re
            # parse saved path from stdout
            m = re.search(r"Model saved to:\s*([^\s]+)", res.stdout)
            cached_path: Optional[Path] = None
            if m:
                cached_path = Path(m.group(1).strip())
            else:
                # fallback: search cache dir for newest .gguf matching repo
                cache_root = Path.home() / ".cache" / "llama.cpp"
                if cache_root.exists():
                    candidates = sorted(cache_root.rglob("*.gguf"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        cached_path = candidates[0]

            if not cached_path or not cached_path.exists():
                # llmfit may have downloaded but not printed path — treat as ok, try to find in cache
                return {"status": "ok", "path": str(d / (filename or cached_path.name if cached_path else "")), "repo_id": repo_id, "filename": filename or (cached_path.name if cached_path else ""), "quant": quant, "output": res.stdout[-1500:]}

            # move/copy from llmfit cache to our app models dir
            target_name = filename or cached_path.name
            dest = d / target_name
            if cached_path.resolve() != dest.resolve():
                try:
                    if dest.exists():
                        dest.unlink()
                    shutil.copy2(str(cached_path), str(dest))
                except Exception as e:
                    raise RuntimeError(f"Copy from cache failed: {e}") from e
            return {"status": "ok", "path": str(dest), "repo_id": repo_id, "filename": target_name, "quant": quant, "output": res.stdout[-1500:]}
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"Download timeout for {repo_id}: {e}") from e
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Download failed: {e}") from e

    def list_remote_ggufs(self, model: str) -> Dict[str, Any]:
        """List available quantizations for a model via `llmfit download <model> --list`."""
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        runner = self._ensure_runner()
        cmd = runner + ["download", model.strip(), "--list"]
        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or res.stdout)
        return {"model": model, "output": res.stdout, "options": res.stdout.splitlines()[:20]}

    def delete_local(self, filename: str) -> Dict[str, Any]:
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")
        filename = filename.strip()
        # prevent path traversal
        if "/" in filename or "\\" in filename or ".." in filename:
            raise ValueError("Invalid filename")
        d = self._models_dir()
        p = d / filename
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        p.unlink()
        return {"status": "deleted", "filename": filename, "path": str(p)}
