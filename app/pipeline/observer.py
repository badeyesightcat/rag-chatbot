import time
import json
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("rag.trace")

# In-memory trace store — last 500 entries, visible in admin UI
TRACE_STORE: list[dict] = []


@contextmanager
def log_phase(phase_name: str, **kwargs):
    """
    Context manager that wraps any pipeline phase.

    Usage:
        with log_phase("retrieval", query=q, top_k=10):
            results = retriever.retrieve(q)

    Automatically records: phase name, duration_ms, status (ok/error), params.
    """
    start = time.perf_counter()
    entry: dict = {"phase": phase_name, "params": kwargs, "status": "running"}
    try:
        yield entry
        entry["status"] = "ok"
    except Exception as e:
        entry["status"] = "error"
        entry["error"]  = str(e)
        raise
    finally:
        entry["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        logger.info(json.dumps(entry))
        TRACE_STORE.append(entry)
        if len(TRACE_STORE) > 500:
            TRACE_STORE.pop(0)


def get_traces(limit: int = 50) -> list[dict]:
    """Returns the most recent traces for the admin dashboard."""
    return TRACE_STORE[-limit:]
