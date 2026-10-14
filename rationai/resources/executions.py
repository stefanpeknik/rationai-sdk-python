from __future__ import annotations

from urllib.parse import quote

from httpx import USE_CLIENT_DEFAULT
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes

from rationai._resource import APIResource, AsyncAPIResource
from rationai.errors import raise_for_api_error
from rationai.types.builders import Execution


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
