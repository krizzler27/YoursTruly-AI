from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import subprocess
import shutil
import json

from config import config
from core.logging import get_logger

logger = get_logger(__name__)

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
        self.models_dir = Path(config.LLAMA_MODEL_PATH)

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

    def _run_ranked(
        self,
        verb: str,
        limit: int,
        sort: str,
        providers: Optional[List[str]],
        perfect_only: bool,
        include_community: bool,
        extra_flags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Shared runner for `fit`/`recommend` - single place for sort/provider/cli shape."""
        from schemas.api_schemas import SORT_ALIASES

        sort = SORT_ALIASES.get(sort.strip().lower(), sort.strip().lower()) if sort else "score"
        if providers is not None and len(providers) == 0:
            providers = None
        runner = self._ensure_runner()
        cmd = runner + [verb, "--json", "--limit", str(limit), "--sort", sort]
        if perfect_only:
            cmd += ["--perfect"]
        if extra_flags:
            cmd += extra_flags
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
        return {"system": system, "raw": raw, "models": models, "catalog": catalog, "flat": flat, "cmd": cmd}

    def discover_and_classify(
        self,
        limit: int = 20,
        include_community: bool = False,
        perfect_only: bool = False,
        providers: Optional[List[str]] = None,
        sort: str = "score",
    ) -> Dict[str, Any]:
        """Run `llmfit fit --json` with filters, return `{system, meta, catalog, models}` ranked for this hardware."""
        r = self._run_ranked("fit", limit, sort, providers, perfect_only, include_community)
        return {
            "system": r["system"],
            "meta": {"cmd": " ".join(r["cmd"]), "limit": limit, "include_community": include_community, "perfect_only": perfect_only, "total": len(r["models"])},
            "catalog": r["catalog"],
            "models": r["flat"],
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
        """Run `llmfit recommend --json` filtered by use_case; providers/sort/perfect filtered client-side."""
        runner = self._ensure_runner()
        cmd = runner + ["recommend", "--json", "--limit", str(limit)]
        if use_case:
            cmd += ["--use-case", use_case]
        if min_fit:
            cmd += ["--min-fit", min_fit]
        if runtime and runtime != "any":
            cmd += ["--runtime", runtime]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW, timeout=30)
        raw = json.loads(res.stdout)
        models = raw.get("models", raw) if isinstance(raw, dict) else raw
        system = raw.get("system") if isinstance(raw, dict) else None
        catalog, flat = self._classify(models)
        # client-side provider filtering (recommend CLI has no --providers)
        if providers is not None and len(providers) > 0:
            wanted = {p.lower() for p in providers}
            flat = [m for m in flat if (m.get("provider") or "").lower() in wanted]
            for k in catalog:
                catalog[k] = [m for m in catalog[k] if (m.get("provider") or "").lower() in wanted]
        elif not include_community and providers is None:
            # trusted-only default for recommend too
            wanted = self.trusted_providers
            flat = [m for m in flat if (m.get("provider") or "").lower() in wanted]
            for k in catalog:
                catalog[k] = [m for m in catalog[k] if (m.get("provider") or "").lower() in wanted]
        if perfect_only:
            flat = [m for m in flat if (m.get("fit_level") or "").lower() == "perfect"]
            catalog = {"perfect": catalog["perfect"], "runnable": [], "others": []}
        # sort client-side (recommend has no --sort)
        from schemas.api_schemas import SORT_ALIASES

        sort = SORT_ALIASES.get(sort.strip().lower(), sort.strip().lower()) if sort else "score"
        if sort == "tps":
            flat.sort(key=lambda m: m.get("tps") or 0, reverse=True)
            for k in catalog:
                catalog[k].sort(key=lambda m: m.get("tps") or 0, reverse=True)
        elif sort == "mem":
            flat.sort(key=lambda m: m.get("vram_gb") or 0)
            for k in catalog:
                catalog[k].sort(key=lambda m: m.get("vram_gb") or 0)
        # score is default order from llmfit (already sorted)
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
        try:
            return sorted([p.name for p in self.models_dir.glob("*.gguf") if p.is_file()])
        except Exception as e:
            logger.warning("list_local failed %s: %s", self.models_dir, e)
            return []

    def list_local_info(self) -> List[Dict[str, object]]:
        """Detailed scan with size for Installed tab."""
        try:
            infos: List[Dict[str, object]] = []
            for p in sorted(self.models_dir.glob("*.gguf"), key=lambda x: x.name.lower()):
                if not p.is_file():
                    continue
                try:
                    st = p.stat()
                    size_gb = round(st.st_size / (1024**3), 2)
                    mtime = st.st_mtime
                except Exception:
                    size_gb = 0
                    mtime = 0
                infos.append({"name": p.name, "path": str(p), "size_gb": size_gb, "size_bytes": st.st_size if 'st' in locals() else 0, "modified": mtime})
            return infos
        except Exception as e:
            logger.warning("list_local_info failed %s: %s", self.models_dir, e)
            return []

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
        p = self.models_dir / filename
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {filename}")
        p.unlink()
        return {"status": "deleted", "filename": filename, "path": str(p)}
