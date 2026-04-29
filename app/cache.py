import redis, json
from app.config import settings

# Single Redis connection shared across all requests
_client = redis.Redis(host=settings.REDIS_HOST, port=6379, decode_responses=True)

CACHE_TTL = 300 # seconds — 5 minutes

def get_cache(key: str) -> dict | None:
    """
    Returns the cached response dict, or None if not found / expired.
    main.py uses this with the walrus operator:
        if cached := get_cache(key): return cached
    """
    try:
        raw = _client.get(key)
        return json.loads(raw) if raw else None
    except redis.RedisError:
        return None # cache miss on any Redis error — degrade gracefully

def set_cache(key: str, value: dict):
    """Store a response dict with TTL. Silently fails if Redis is down."""
    try:
        _client.setex(key, CACHE_TTL, json.dumps(value))
    except redis.RedisError:
        pass # cache write failure is non-fatal — app still works

def clear_cache():
    """Wipe all cached responses. Called from admin UI after re-indexing documents."""
    try:
        _client.flushdb()
    except redis.RedisError:
        pass