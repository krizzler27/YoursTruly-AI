from typing import Any, Dict, List, Optional, Set
import subprocess
import shutil
import json
import httpx
import logging

from config import config

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

DEFAULT_TRUSTED_PROVIDERS: Set[str] = {
    "meta",
    "llama",
    "google",
    "gemma",
    "qwen",
    "alibaba",
    "deepseek",
    "mistral",
    "microsoft",
    "phi",
}

class LLMFitServices:
    """Hardware-aware model discovery for local inference."""

    def __init__(
        self,
        trusted_providers: Optional[Set[str]] = None,
        ollama_host: str = config.DEFAULT_OLLAMA_HOST,
    ):
        self.trusted_providers = {p.lower() for p in (trusted_providers or DEFAULT_TRUSTED_PROVIDERS)}
        self.ollama_host = ollama_host.rstrip("/")

    def _ensure_runner(self) -> List[str]:
        if shutil.which("llmfit"):
            return ["llmfit"]
        if shutil.which("uvx"):
            return ["uvx", "llmfit"]
        raise FileNotFoundError("llmfit not found: scoop install llmfit / brew install llmfit")

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
            item = {
                "name": m.get("name"),
                "provider": m.get("provider"),
                "fit_level": m.get("fit_level"),
                "quant": m.get("best_quant"),
                "run_mode": m.get("run_mode"),
                "ollama_name": m.get("ollama_name"),
                "vram_gb": m.get("memory_required_gb"),
                "tps": m.get("estimated_tps"),
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

    async def list_installed(self, exclude_embed: bool = True, embed_only: bool = False, timeout: float = 2.0) -> List[str]:
        if not await self.is_ollama_running(timeout=timeout):
            raise Exception("Ollama is not running. Run 'ollama serve' in terminal")

        try:
            names = []
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{self.ollama_host}/api/tags")
                r.raise_for_status()
                data = r.json()
                names = [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception as e:
            logger.warning("list_installed failed (%s): %s", self.ollama_host, e)
            return []

        if exclude_embed:
            return [name for name in names if "embed" not in name.lower()]

        if embed_only:
            return [n for n in names if "embed" in n.lower()]
        return names

    def pull(self, model: str) -> Dict[str, Any]:
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        model = model.strip()
        if not shutil.which("ollama"):
            raise FileNotFoundError("Ollama not installed: https://ollama.com/download")
        res = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW, timeout=600)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or res.stdout)
        return {"status": "ok", "method": "ollama", "model": model, "output": res.stdout[-1500:]}

    async def is_ollama_running(self, timeout: float = 2.0) -> bool:
        """Async check for Ollama status."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(self.ollama_host)
                return response.status_code == 200
        except httpx.RequestError:
            return False