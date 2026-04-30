import json
import redis
from app.config import settings

_client = redis.Redis(host=settings.REDIS_HOST, port=6379, decode_responses=True)

CACHE_TTL = 300   # seconds — 5 minutes


def get_cache(key: str) -> dict | None:
    try:
        raw = _client.get(key)
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None   # degrade gracefully — cache miss on any Redis error


def set_cache(key: str, value: dict):
    try:
        _client.setex(key, CACHE_TTL, json.dumps(value))
    except redis.RedisError:
        pass   # non-fatal — app still works without cache


def clear_cache():
    """Wipe all cached responses. Call after re-indexing documents."""
    try:
        _client.flushdb()
    except redis.RedisError:
        pass
