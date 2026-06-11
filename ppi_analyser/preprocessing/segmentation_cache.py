# preprocessing/segmentation_cache.py

import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_PATH = Path.home() / ".ppi_analyser" / "segmentation_cache.json"
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if _CACHE_PATH.exists():
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            logger.info("Segmentation cache loaded: %d entries from %s", len(_cache), _CACHE_PATH)
        except Exception as e:
            logger.warning("Could not load segmentation cache: %s — starting fresh", e)
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("Could not save segmentation cache: %s", e)


def _key(text: str, model: str) -> str:
    """Cache key = md5(text + model) so different models get different entries."""
    return hashlib.md5((text + model).encode("utf-8")).hexdigest()


def get(text: str, model: str) -> str | None:
    cache = _load()
    result = cache.get(_key(text, model))
    if result is not None:
        logger.debug("Segmentation cache HIT for text: %.60s...", text)
    return result


def set(text: str, model: str, result: str) -> None:
    cache = _load()
    cache[_key(text, model)] = result
    _save()


def cache_size() -> int:
    return len(_load())


def clear() -> None:
    global _cache
    _cache = {}
    _save()
    logger.info("Segmentation cache cleared")
