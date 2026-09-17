from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import quote
from uuid import uuid4

import httpx
from httpx import USE_CLIENT_DEFAULT
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes

from rationai._resource import APIResource, AsyncAPIResource
from rationai.errors import (
    APIErrorCode,
    SchedulingUnavailableError,
    api_error,
    raise_for_api_error,
)
from rationai.types.workflows import ScheduledExecution, WorkflowResult


if TYPE_CHECKING:
    from rationai.types.workflows import WorkflowRequest


_SCHEDULE_ATTEMPTS = 3


class Workflows(APIResource):
    def run(
        self,
        workflow_id: str,
        request: WorkflowRequest,
        *,
        workflow_timeout_seconds: float = 3600,
        timeout: TimeoutTypes | UseClientDefault = None,
    ) -> WorkflowResult:
        body = _run_body(request, workflow_timeout_seconds)
        response = self._post(
            f"/v1/workflows/{quote(workflow_id, safe='')}/run",
            json=body,
            timeout=timeout,
        )
        raise_for_api_error(response)
        return WorkflowResult.from_dict(response.json())

    def schedule(
        self,
        workflow_id: str,
        request: WorkflowRequest,
        *,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> ScheduledExecution:
        key = idempotency_key or str(uuid4())
        body = _schedule_body(request, requested_timeout_seconds, key)
        for attempt in range(_SCHEDULE_ATTEMPTS):
            try:
                response = self._post(
                    f"/v1/workflows/{quote(workflow_id, safe='')}/schedule",
                    json=body,
                    timeout=timeout,
                )
            except httpx.TransportError as error:
                if attempt + 1 == _SCHEDULE_ATTEMPTS:
                    raise _transport_unavailable() from error
                time.sleep(_retry_delay(None, attempt))
                continue
            if _scheduling_unavailable(response) and attempt + 1 < _SCHEDULE_ATTEMPTS:
                time.sleep(_retry_delay(response, attempt))
                continue
            raise_for_api_error(response)
            return ScheduledExecution.from_dict(
                response.json(),
                idempotency_key=key,
            )
        raise AssertionError("schedule retry loop did not return or raise")


class AsyncWorkflows(AsyncAPIResource):
    async def run(
        self,
        workflow_id: str,
        request: WorkflowRequest,
        *,
        workflow_timeout_seconds: float = 3600,
        timeout: TimeoutTypes | UseClientDefault = None,
    ) -> WorkflowResult:
        body = _run_body(request, workflow_timeout_seconds)
        response = await self._post(
            f"/v1/workflows/{quote(workflow_id, safe='')}/run",
            json=body,
            timeout=timeout,
        )
        raise_for_api_error(response)
        return WorkflowResult.from_dict(response.json())

    async def schedule(
        self,
        workflow_id: str,
        request: WorkflowRequest,
        *,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> ScheduledExecution:
        key = idempotency_key or str(uuid4())
        body = _schedule_body(request, requested_timeout_seconds, key)
        for attempt in range(_SCHEDULE_ATTEMPTS):
            try:
                response = await self._post(
                    f"/v1/workflows/{quote(workflow_id, safe='')}/schedule",
                    json=body,
                    timeout=timeout,
                )
            except httpx.TransportError as error:
                if attempt + 1 == _SCHEDULE_ATTEMPTS:
                    raise _transport_unavailable() from error
                await asyncio.sleep(_retry_delay(None, attempt))
                continue
            if _scheduling_unavailable(response) and attempt + 1 < _SCHEDULE_ATTEMPTS:
                await asyncio.sleep(_retry_delay(response, attempt))
                continue
            raise_for_api_error(response)
            return ScheduledExecution.from_dict(
                response.json(),
                idempotency_key=key,
            )
        raise AssertionError("schedule retry loop did not return or raise")


def _run_body(
    request: WorkflowRequest,
    workflow_timeout_seconds: float,
) -> dict[str, object]:
    if (
        not isinstance(workflow_timeout_seconds, int | float)
        or isinstance(workflow_timeout_seconds, bool)
        or workflow_timeout_seconds <= 0
    ):
        raise ValueError("workflow_timeout_seconds must be positive")
    return {
        **request.to_dict(),
        "deadline_at": (
            datetime.now(UTC) + timedelta(seconds=workflow_timeout_seconds)
        ).isoformat(),
    }


def _schedule_body(
    request: WorkflowRequest,
    requested_timeout_seconds: int | None,
    idempotency_key: str,
) -> dict[str, object]:
    if requested_timeout_seconds is not None and (
        not isinstance(requested_timeout_seconds, int)
        or isinstance(requested_timeout_seconds, bool)
        or requested_timeout_seconds <= 0
    ):
        raise ValueError("requested_timeout_seconds must be a positive integer")
    if not idempotency_key:
        raise ValueError("idempotency_key must be non-empty")
    return {
        "provider": "standalone",
        "body": request.to_dict(),
        "requested_timeout_seconds": requested_timeout_seconds,
        "idempotency_key": idempotency_key,
    }


def _scheduling_unavailable(response: httpx.Response) -> bool:
    return isinstance(api_error(response), SchedulingUnavailableError)


def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        value = response.headers.get("retry-after")
        if value is not None:
            try:
                return max(0.0, min(float(value), 2.0))
            except ValueError:
                pass
    return min(0.25 * (2**attempt), 2.0)


def _transport_unavailable() -> SchedulingUnavailableError:
    return SchedulingUnavailableError(
        "execution scheduling is temporarily unavailable",
        code=APIErrorCode.SCHEDULING_UNAVAILABLE,
        status_code=None,
        retryable=True,
    )
