"""Retry policy shared by every durable submission (builders and raw tiles)."""

from __future__ import annotations

import httpx

from rationai.errors import APIErrorCode, SchedulingUnavailableError, api_error


SCHEDULE_ATTEMPTS = 3


def scheduling_unavailable(response: httpx.Response) -> bool:
    return isinstance(api_error(response), SchedulingUnavailableError)


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        value = response.headers.get("retry-after")
        if value is not None:
            try:
                return max(0.0, min(float(value), 2.0))
            except ValueError:
                pass
    return min(0.25 * (2**attempt), 2.0)


def transport_unavailable() -> SchedulingUnavailableError:
    return SchedulingUnavailableError(
        "execution scheduling is temporarily unavailable",
        code=APIErrorCode.SCHEDULING_UNAVAILABLE,
        status_code=None,
        retryable=True,
    )


def check_timeout(requested_timeout_seconds: int | None) -> None:
    if requested_timeout_seconds is not None and (
        not isinstance(requested_timeout_seconds, int)
        or isinstance(requested_timeout_seconds, bool)
        or requested_timeout_seconds <= 0
    ):
        raise ValueError("requested_timeout_seconds must be a positive integer")
