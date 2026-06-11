# analysis/analysis_cache.py

import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_cache: dict | None = None
_cache_path: Path | None = None


def init(cache_path: str) -> None:
    """Call once at the start of a run to set the cache path and load existing entries."""
    global _cache, _cache_path
    _cache_path = Path(cache_path)
    if _cache_path.exists():
        try:
            _cache = json.loads(_cache_path.read_text(encoding="utf-8"))
            logger.info("Analysis cache loaded: %d entries from %s", len(_cache), _cache_path)
        except Exception as e:
            logger.warning("Could not load analysis cache: %s — starting fresh", e)
            _cache = {}
    else:
        _cache = {}


def _require_init() -> None:
    if _cache is None:
        raise RuntimeError(
            "Analysis cache not initialised — call analysis_cache.init(path) first, "
            "or set use_analysis_cache=True and analysis_cache_path in PipelineConfig."
        )


def _key(conversation: str, expression: str, model: str, submodel: str, prompt_type: str) -> str:
    payload = f"{conversation}|{expression}|{model}|{submodel}|{prompt_type}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def get(conversation: str, expression: str, model: str, submodel: str, prompt_type: str) -> str | None:
    _require_init()
    result = _cache.get(_key(conversation, expression, model, submodel, prompt_type))
    if result is not None:
        logger.debug("Analysis cache HIT — %s / %s / %s", prompt_type, model, expression[:40])
    return result


def set(conversation: str, expression: str, model: str, submodel: str, prompt_type: str, result: str) -> None:
    _require_init()
    _cache[_key(conversation, expression, model, submodel, prompt_type)] = result
    _save()


def _save() -> None:
    try:
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        _cache_path.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save analysis cache: %s", e)


def cache_size() -> int:
    _require_init()
    return len(_cache)


def clear() -> None:
    global _cache
    _require_init()
    _cache = {}
    _save()
    logger.info("Analysis cache cleared")
