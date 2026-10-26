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
from rationai.errors import raise_for_api_error
from rationai.resources._scheduling import (
    SCHEDULE_ATTEMPTS,
    check_timeout,
    retry_delay,
    scheduling_unavailable,
    transport_unavailable,
)
from rationai.types.builders import BuilderInfo, BuilderResult, ScheduledExecution


if TYPE_CHECKING:
    from rationai.types.builders import BuilderRequest


class Builders(APIResource):
    def list(
        self, *, timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT
    ) -> list[BuilderInfo]:
        """List builders with the inference contract each accepts."""
        response = self._get("v1/builders", timeout=timeout)
        raise_for_api_error(response)
        return [BuilderInfo.from_dict(item) for item in response.json()]

    def run(
        self,
        builder_id: str,
        request: BuilderRequest,
        *,
        run_timeout_seconds: float = 3600,
        timeout: TimeoutTypes | UseClientDefault = None,
    ) -> BuilderResult:
        """Run a builder with the request's model and wait for its outputs.

        Args:
            builder_id: Builder such as "heatmap" or "tile-embeddings".
            request: Selected model, inputs and input access.
            run_timeout_seconds: Server-side deadline from the time of this call.
            timeout: HTTP timeout; defaults to no timeout for this request.

        Returns:
            The builder outputs.
        """
        body = _run_body(request, run_timeout_seconds)
        response = self._post(
            _path(builder_id, request, "run"),
            json=body,
            timeout=timeout,
        )
        raise_for_api_error(response)
        return BuilderResult.from_dict(response.json())

    def schedule(
        self,
        builder_id: str,
        request: BuilderRequest,
        *,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> ScheduledExecution:
        """Submit durable work, reusing the same key during transient retries.

        Args:
            builder_id: Builder such as "heatmap" or "tile-embeddings".
            request: Selected model, inputs and input access.
            requested_timeout_seconds: Optional server-side execution deadline.
            idempotency_key: Recovery key; generated if omitted. Reuse the same
                key after a client restart to recover an ambiguous submission.
            timeout: HTTP timeout for each submission attempt.

        Returns:
            The admitted execution, replay flag, and idempotency key.
        """
        key = idempotency_key or str(uuid4())
        body = _schedule_body(request, requested_timeout_seconds)
        headers = _schedule_headers(key)
        for attempt in range(SCHEDULE_ATTEMPTS):
            try:
                response = self._post(
                    _path(builder_id, request, "schedule"),
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            except httpx.TransportError as error:
                if attempt + 1 == SCHEDULE_ATTEMPTS:
                    raise transport_unavailable() from error
                time.sleep(retry_delay(None, attempt))
                continue
            if scheduling_unavailable(response) and attempt + 1 < SCHEDULE_ATTEMPTS:
                time.sleep(retry_delay(response, attempt))
                continue
            raise_for_api_error(response)
            return ScheduledExecution.from_dict(
                response.json(),
                idempotency_key=key,
            )
        raise AssertionError("schedule retry loop did not return or raise")


class AsyncBuilders(AsyncAPIResource):
    async def list(
        self, *, timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT
    ) -> list[BuilderInfo]:
        """List builders with the inference contract each accepts."""
        response = await self._get("v1/builders", timeout=timeout)
        raise_for_api_error(response)
        return [BuilderInfo.from_dict(item) for item in response.json()]

    async def run(
        self,
        builder_id: str,
        request: BuilderRequest,
        *,
        run_timeout_seconds: float = 3600,
        timeout: TimeoutTypes | UseClientDefault = None,
    ) -> BuilderResult:
        """Run a builder with the request's model and wait for its outputs.

        Args:
            builder_id: Builder such as "heatmap" or "tile-embeddings".
            request: Selected model, inputs and input access.
            run_timeout_seconds: Server-side deadline from the time of this call.
            timeout: HTTP timeout; defaults to no timeout for this request.

        Returns:
            The builder outputs.
        """
        body = _run_body(request, run_timeout_seconds)
        response = await self._post(
            _path(builder_id, request, "run"),
            json=body,
            timeout=timeout,
        )
        raise_for_api_error(response)
        return BuilderResult.from_dict(response.json())

    async def schedule(
        self,
        builder_id: str,
        request: BuilderRequest,
        *,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> ScheduledExecution:
        """Submit durable work, reusing the same key during transient retries.

        Args:
            builder_id: Builder such as "heatmap" or "tile-embeddings".
            request: Selected model, inputs and input access.
            requested_timeout_seconds: Optional server-side execution deadline.
            idempotency_key: Recovery key; generated if omitted. Reuse the same
                key after a client restart to recover an ambiguous submission.
            timeout: HTTP timeout for each submission attempt.

        Returns:
            The admitted execution, replay flag, and idempotency key.
        """
        key = idempotency_key or str(uuid4())
        body = _schedule_body(request, requested_timeout_seconds)
        headers = _schedule_headers(key)
        for attempt in range(SCHEDULE_ATTEMPTS):
            try:
                response = await self._post(
                    _path(builder_id, request, "schedule"),
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
            except httpx.TransportError as error:
                if attempt + 1 == SCHEDULE_ATTEMPTS:
                    raise transport_unavailable() from error
                await asyncio.sleep(retry_delay(None, attempt))
                continue
            if scheduling_unavailable(response) and attempt + 1 < SCHEDULE_ATTEMPTS:
                await asyncio.sleep(retry_delay(response, attempt))
                continue
            raise_for_api_error(response)
            return ScheduledExecution.from_dict(
                response.json(),
                idempotency_key=key,
            )
        raise AssertionError("schedule retry loop did not return or raise")


def _run_body(
    request: BuilderRequest,
    run_timeout_seconds: float,
) -> dict[str, object]:
    if (
        not isinstance(run_timeout_seconds, int | float)
        or isinstance(run_timeout_seconds, bool)
        or run_timeout_seconds <= 0
    ):
        raise ValueError("run_timeout_seconds must be positive")
    return {
        **request.to_dict(),
        "deadline_at": (
            datetime.now(UTC) + timedelta(seconds=run_timeout_seconds)
        ).isoformat(),
    }


def _schedule_body(
    request: BuilderRequest,
    requested_timeout_seconds: int | None,
) -> dict[str, object]:
    check_timeout(requested_timeout_seconds)
    return {
        **request.to_dict(),
        "requested_timeout_seconds": requested_timeout_seconds,
    }


def _schedule_headers(idempotency_key: str) -> dict[str, str]:
    if not idempotency_key:
        raise ValueError("idempotency_key must be non-empty")
    return {"idempotency-key": idempotency_key}


def _path(builder_id: str, request: BuilderRequest, action: str) -> str:
    builder = quote(builder_id, safe="")
    model = quote(request.model_id, safe="")
    return f"v1/builders/{builder}/models/{model}/{action}"
