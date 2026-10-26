from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

from httpx import USE_CLIENT_DEFAULT
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes

from rationai._resource import APIResource, AsyncAPIResource
from rationai.errors import raise_for_api_error
from rationai.types.builders import Execution
from rationai.types.inference import InferenceResponse


if TYPE_CHECKING:
    from collections.abc import Sequence

    import httpx


LOOKUP_LIMIT = 1000


class Executions(APIResource):
    def get(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Execution:
        response = self._get(
            f"v1/executions/{quote(execution_id, safe='')}",
            timeout=timeout,
        )
        raise_for_api_error(response)
        return Execution.from_dict(response.json())

    def cancel(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Execution:
        response = self._post(
            f"v1/executions/{quote(execution_id, safe='')}/cancel",
            timeout=timeout,
        )
        raise_for_api_error(response)
        return Execution.from_dict(response.json())

    def lookup(
        self,
        execution_ids: Sequence[str],
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> list[Execution]:
        """Fetch many executions at once; unknown IDs are left out."""
        found: list[Execution] = []
        for start in range(0, len(execution_ids), LOOKUP_LIMIT):
            response = self._post(
                "v1/executions/lookup",
                json={"ids": list(execution_ids[start : start + LOOKUP_LIMIT])},
                timeout=timeout,
            )
            raise_for_api_error(response)
            found.extend(Execution.from_dict(item) for item in response.json())
        return found

    def wait(
        self,
        execution_ids: Sequence[str],
        *,
        poll_interval: float = 2.0,
        wait_timeout: float | None = None,
    ) -> list[Execution]:
        """Poll until every listed execution has finished, in the given order.

        Raises TimeoutError after `wait_timeout` seconds.
        """
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        while True:
            executions = self.lookup(execution_ids)
            if len(executions) == len(execution_ids) and all(
                execution.finished for execution in executions
            ):
                return executions
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("executions did not finish in time")
            time.sleep(poll_interval)

    def result(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> InferenceResponse:
        """Download a succeeded scheduled tile's model response."""
        response = self._get(
            f"v1/executions/{quote(execution_id, safe='')}/result", timeout=timeout
        )
        return _inference_response(response)


class AsyncExecutions(AsyncAPIResource):
    async def get(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Execution:
        response = await self._get(
            f"v1/executions/{quote(execution_id, safe='')}",
            timeout=timeout,
        )
        raise_for_api_error(response)
        return Execution.from_dict(response.json())

    async def cancel(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> Execution:
        response = await self._post(
            f"v1/executions/{quote(execution_id, safe='')}/cancel",
            timeout=timeout,
        )
        raise_for_api_error(response)
        return Execution.from_dict(response.json())

    async def lookup(
        self,
        execution_ids: Sequence[str],
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> list[Execution]:
        """Fetch many executions at once; unknown IDs are left out."""
        found: list[Execution] = []
        for start in range(0, len(execution_ids), LOOKUP_LIMIT):
            response = await self._post(
                "v1/executions/lookup",
                json={"ids": list(execution_ids[start : start + LOOKUP_LIMIT])},
                timeout=timeout,
            )
            raise_for_api_error(response)
            found.extend(Execution.from_dict(item) for item in response.json())
        return found

    async def wait(
        self,
        execution_ids: Sequence[str],
        *,
        poll_interval: float = 2.0,
        wait_timeout: float | None = None,
    ) -> list[Execution]:
        """Poll until every listed execution has finished, in the given order.

        Raises TimeoutError after `wait_timeout` seconds.
        """
        deadline = None if wait_timeout is None else time.monotonic() + wait_timeout
        while True:
            executions = await self.lookup(execution_ids)
            if len(executions) == len(execution_ids) and all(
                execution.finished for execution in executions
            ):
                return executions
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("executions did not finish in time")
            await asyncio.sleep(poll_interval)

    async def result(
        self,
        execution_id: str,
        *,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> InferenceResponse:
        """Download a succeeded scheduled tile's model response."""
        response = await self._get(
            f"v1/executions/{quote(execution_id, safe='')}/result", timeout=timeout
        )
        return _inference_response(response)


def _inference_response(response: httpx.Response) -> InferenceResponse:
    raise_for_api_error(response)
    return InferenceResponse(
        content=response.content,
        media_type=response.headers.get("content-type"),
        headers={
            name: value
            for name, value in response.headers.items()
            if name.startswith("x-")
        },
    )
