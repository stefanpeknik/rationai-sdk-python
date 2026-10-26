from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self

import httpx
from httpx import URL
from httpx._types import TimeoutTypes


if TYPE_CHECKING:
    from rationai.resources.builders import AsyncBuilders, Builders
    from rationai.resources.executions import AsyncExecutions, Executions
    from rationai.resources.models import AsyncModels, Models
    from rationai.resources.qc import AsyncQualityControl, QualityControl
    from rationai.resources.slide import AsyncSlide, Slide


DEFAULT_MODEL_SERVICE_URL = (
    "http://rayservice-model-execution-scheduler."
    "rationai-jobs-ns.svc.cluster.local:8080"
)


def _model_service_url(
    base_url: URL | str | None, models_base_url: URL | str | None
) -> URL | str:
    if base_url is None:
        return (
            models_base_url
            if models_base_url is not None
            else DEFAULT_MODEL_SERVICE_URL
        )
    if models_base_url is not None and URL(base_url) != URL(models_base_url):
        raise ValueError("base_url and models_base_url must refer to the same service")
    return base_url


class Client(httpx.Client):
    def __init__(
        self,
        *,
        base_url: URL | str | None = None,
        models_base_url: URL | str | None = None,
        qc_base_url: URL
        | str = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        timeout: TimeoutTypes = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=_model_service_url(base_url, models_base_url),
            timeout=timeout,
            follow_redirects=True,
            **kwargs,
        )
        self.qc_base_url = qc_base_url

    def __enter__(self) -> Self:
        super().__enter__()
        return self

    @cached_property
    def models(self) -> Models:
        from rationai.resources.models import Models

        return Models(self, base_url=self.base_url)

    @cached_property
    def slide(self) -> Slide:
        from rationai.resources.slide import Slide

        return Slide(self, base_url=self.base_url)

    @cached_property
    def qc(self) -> QualityControl:
        from rationai.resources.qc import QualityControl

        return QualityControl(self, base_url=self.qc_base_url)

    @cached_property
    def builders(self) -> Builders:
        from rationai.resources.builders import Builders

        return Builders(self, base_url=self.base_url)

    @cached_property
    def executions(self) -> Executions:
        from rationai.resources.executions import Executions

        return Executions(self, base_url=self.base_url)


class AsyncClient(httpx.AsyncClient):
    def __init__(
        self,
        *,
        base_url: URL | str | None = None,
        models_base_url: URL | str | None = None,
        qc_base_url: URL
        | str = "http://rayservice-qc-serve-svc.rationai-jobs-ns.svc.cluster.local:8000",
        timeout: TimeoutTypes = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=_model_service_url(base_url, models_base_url),
            timeout=timeout,
            follow_redirects=True,
            **kwargs,
        )
        self.qc_base_url = qc_base_url

    async def __aenter__(self) -> Self:
        await super().__aenter__()
        return self

    @cached_property
    def models(self) -> AsyncModels:
        from rationai.resources.models import AsyncModels

        return AsyncModels(self, base_url=self.base_url)

    @cached_property
    def slide(self) -> AsyncSlide:
        from rationai.resources.slide import AsyncSlide

        return AsyncSlide(self, base_url=self.base_url)

    @cached_property
    def qc(self) -> AsyncQualityControl:
        from rationai.resources.qc import AsyncQualityControl

        return AsyncQualityControl(self, base_url=self.qc_base_url)

    @cached_property
    def builders(self) -> AsyncBuilders:
        from rationai.resources.builders import AsyncBuilders

        return AsyncBuilders(self, base_url=self.base_url)

    @cached_property
    def executions(self) -> AsyncExecutions:
        from rationai.resources.executions import AsyncExecutions

        return AsyncExecutions(self, base_url=self.base_url)
