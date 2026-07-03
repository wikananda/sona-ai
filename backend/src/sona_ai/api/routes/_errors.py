from contextlib import contextmanager

import httpx
from fastapi import HTTPException

from sona_ai.core import setup_logging

logger = setup_logging()


@contextmanager
def route_error_handler(
    log_message: str,
    byok_errors: bool = False,
    log_traceback: bool = True,
):
    """Map route exceptions to HTTP errors.

    ValueError becomes a 400, everything else is logged and becomes a 500.
    With `byok_errors=True`, httpx.HTTPStatusError becomes a 400 (with a
    friendly message for 401 responses). `log_message` must contain a single
    `%s` placeholder for the error text.
    """
    try:
        yield
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        if byok_errors and isinstance(exc, httpx.HTTPStatusError):
            if exc.response.status_code == 401:
                raise HTTPException(
                    status_code=400,
                    detail="BYOK authentication failed. Check provider, API key, and model.",
                ) from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if log_traceback:
            logger.exception(log_message, str(exc))
        else:
            logger.error(log_message, str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc
