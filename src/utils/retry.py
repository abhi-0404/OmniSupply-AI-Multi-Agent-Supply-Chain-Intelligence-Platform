"""
Exponential backoff retry utility for Gemini API rate limit errors.

Usage:
    from src.utils.retry import gemini_retry

    result = gemini_retry(lambda: llm.invoke(prompt), max_retries=3, base_delay=5)
"""

import time
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Signals that identify a Gemini rate-limit / quota error
_RATE_LIMIT_SIGNALS = (
    "RESOURCE_EXHAUSTED",
    "429",
    "quota",
    "rate limit",
    "Too Many Requests",
)


def _is_rate_limit(exc: Exception) -> bool:
    """Return True if the exception is a Gemini quota / rate-limit error."""
    msg = str(exc).lower()
    return any(signal.lower() in msg for signal in _RATE_LIMIT_SIGNALS)


def gemini_retry(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> T:
    """
    Call fn() and retry on Gemini rate-limit errors with exponential backoff.

    Backoff schedule (base_delay=5):
        attempt 1 → wait  5s
        attempt 2 → wait 10s
        attempt 3 → wait 20s
        give up   → re-raise original exception

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts (not counting the first call).
        base_delay:  Base wait time in seconds; doubles each attempt.

    Returns:
        The return value of fn() on success.

    Raises:
        The original exception after max_retries are exhausted.
    """
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc

            if not _is_rate_limit(exc):
                # Not a rate-limit error — don't retry
                raise

            if attempt >= max_retries:
                logger.error(
                    f"Rate limit: giving up after {max_retries} retries. "
                    f"Last error: {str(exc)[:120]}"
                )
                raise

            wait = base_delay * (2 ** attempt)
            logger.warning(
                f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                f"Retrying in {wait:.0f}s... [{str(exc)[:80]}]"
            )
            time.sleep(wait)

    # Should never reach here, but satisfies type checker
    raise last_exc  # type: ignore[misc]
