# =============================================================
# PHASE 6: OBSERVABILITY
# A context manager that wraps each pipeline phase.
# Automatically logs: phase name, duration, key params, errors.
# Output goes to: console + in-memory trace store (shown in UI)
# =============================================================

import time, json, logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("rag.trace")

# In-memory trace store — last 100 requests visible in admin UI
TRACE_STORE: list[dict] = []

@contextmanager
def log_phase(phase_name: str, **kwargs):
    """
    Usage:
        with log_phase("retrieval", query=q, top_k=10):
            results = retriever.retrieve(q)
    """
    start = time.perf_counter()
    entry = {
        "phase": phase_name,
        "params": kwargs,
        "status": "running"
    }
    try:
        yield entry
        entry["status"] = "ok"
    except Exception as e:
        entry["status"] = "error"
        entry["error"] = str(e)
        raise
    finally:
        entry["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info(json.dumps(entry))
        TRACE_STORE.append(entry)
        if len(TRACE_STORE) > 500:
            TRACE_STORE.pop(0) # keep recent only

# API endpoint to get traces for admin dashboard
def get_traces(limit=50) -> list[dict]:
    return TRACE_STORE[-limit:]