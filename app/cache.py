import json
import redis
from app.config import settings

_client = redis.Redis(host=settings.REDIS_HOST, port=6379, decode_responses=True)

CACHE_TTL = 300   # seconds — 5 minutes


def get_cache(key: str) -> dict | None:
    """
    Returns the cached response dict, or None on any failure.
    Catching Exception (not just RedisError) ensures the app never
    crashes due to cache issues regardless of connection state.
    """
    try:
        raw = _client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None   # degrade gracefully — treat any error as a cache miss


def set_cache(key: str, value: dict):
    """Store a response dict with TTL. Silently no-ops if Redis is unavailable."""
    try:
        _client.setex(key, CACHE_TTL, json.dumps(value))
    except Exception:
        pass   # non-fatal — app still works without cache


def clear_cache():
    """Wipe all cached responses. Call after re-indexing documents."""
    try:
        _client.flushdb()
    except Exception:
        pass
