import asyncio
import time
from typing import cast
from uuid import uuid4

import httpx
import lz4.frame
import numpy as np
from httpx import USE_CLIENT_DEFAULT
from httpx._client import UseClientDefault
from httpx._types import TimeoutTypes
from numpy.typing import NDArray
from PIL.Image import Image

from rationai._resource import APIResource, AsyncAPIResource
from rationai.errors import raise_for_api_error
from rationai.resources._scheduling import (
    SCHEDULE_ATTEMPTS,
    retry_delay,
    scheduling_unavailable,
    transport_unavailable,
)
from rationai.resources._tiles import schedule_request
from rationai.types.builders import ModelInfo, ScheduledExecution


class Models(APIResource):
    def list(
        self, *, timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT
    ) -> list[ModelInfo]:
        """List deployed models with their contracts and compatible builders."""
        response = self._get("v1/models", timeout=timeout)
        raise_for_api_error(response)
        return [ModelInfo.from_dict(item) for item in response.json()]

    def classify_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> float | dict[str, float]:
        """Classify an image using the specified model.

        Args:
            model: The name of the model to use for classification.
            image: The image to classify. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            (float | dict[str, float]): The classification result as a single float
                (for binary classification) or probabilities for each class.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    def segment_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[np.float16]:
        """Segment an image using the specified model.

        Args:
            model: The name of the model to use for segmentation.
            image: The image to segment. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            NDArray[np.float16]: The segmentation result as a numpy array of float16 values.
                The shape of the array is (num_classes, height, width).
        """
        if isinstance(image, Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]

        compressed_data = lz4.frame.compress(image.tobytes())
        response = self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)

    def embed_image[DType: np.generic](
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        output_dtype: type[DType] = np.float32,  # type: ignore[assignment]
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        **headers: str,
    ) -> NDArray[DType]:
        """Compute an embedding vector for an image using the specified model.

        Args:
            model: The name of the model to use for embedding.
            image: The image to embed. It must be uint8 RGB image.
            output_dtype: Output numpy dtype for embeddings (e.g. np.float16, np.float32).
            timeout: Optional timeout for the request.
            **headers: Additional x- headers. Keyword underscores are converted
                to hyphens and prefixed with 'x-', e.g. pool_tokens="false"
                becomes x-pool-tokens: false.

        Returns:
            NDArray[DType]: The embedding array reshaped according to
                the `x-output-shape` response header.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        request_headers = {"x-output-dtype": np.dtype(output_dtype).name}
        request_headers.update(
            {f"x-{k.replace('_', '-')}": v for k, v in headers.items()}
        )

        response = self._post(
            model,
            data=compressed_data,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = lz4.frame.decompress(response.content)
        embedding = np.frombuffer(payload, dtype=output_dtype)

        response_shape = response.headers["x-output-shape"]
        embedding = embedding.reshape(eval(response_shape))

        return cast("NDArray[DType]", embedding)

    def schedule(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        *,
        output_dtype: type[np.generic] | None = None,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        **headers: str,
    ) -> ScheduledExecution:
        """Queue one tile for the model instead of waiting on the connection.

        The tile must be what the model's raw endpoint takes: a uint8 RGB image
        of its `tile_size`. Collect the outcome with `executions.wait` and
        `executions.result`; the response is kept for a limited time.

        Args:
            model: Model ID.
            image: The tile; must be a uint8 RGB image of the model's tile size.
            output_dtype: Output dtype for models that support `x-output-dtype`.
            requested_timeout_seconds: Optional server-side deadline.
            idempotency_key: Recovery key; generated if omitted. Reuse the same
                key after a client restart to recover an ambiguous submission.
            timeout: HTTP timeout for each submission attempt.
            **headers: Additional x- headers, e.g. pool_tokens="false".

        Returns:
            The admitted execution, replay flag, and idempotency key.
        """
        key = idempotency_key or str(uuid4())
        request = schedule_request(
            model,
            image,
            idempotency_key=key,
            output_dtype=output_dtype,
            requested_timeout_seconds=requested_timeout_seconds,
            headers=headers,
        )
        for attempt in range(SCHEDULE_ATTEMPTS):
            try:
                response = self._post(
                    request.url,
                    content=request.content,
                    headers=request.headers,
                    params=request.params,
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
            return ScheduledExecution.from_dict(response.json(), idempotency_key=key)
        raise AssertionError("schedule retry loop did not return or raise")


class AsyncModels(AsyncAPIResource):
    async def list(
        self, *, timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT
    ) -> list[ModelInfo]:
        """List deployed models with their contracts and compatible builders."""
        response = await self._get("v1/models", timeout=timeout)
        raise_for_api_error(response)
        return [ModelInfo.from_dict(item) for item in response.json()]

    async def classify_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> float | dict[str, float]:
        """Classify an image using the specified model.

        Args:
            model: The name of the model to use for classification.
            image: The image to classify. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            (float | dict[str, float]): The classification result as a single float
                (for binary classification) or probabilities for each class.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        response = await self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def segment_image(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
    ) -> NDArray[np.float16]:
        """Segment an image using the specified model.

        Args:
            model: The name of the model to use for segmentation.
            image: The image to segment. It must be uint8 RGB image.
            timeout: Optional timeout for the request.

        Returns:
            NDArray[np.float16]: The segmentation result as a numpy array of float16 values.
                The shape of the array is (num_classes, height, width).
        """
        if isinstance(image, Image):
            w, h = image.size
        else:
            h, w = image.shape[:2]

        compressed_data = lz4.frame.compress(image.tobytes())
        response = await self._post(model, data=compressed_data, timeout=timeout)
        response.raise_for_status()

        return np.frombuffer(
            lz4.frame.decompress(response.content), dtype=np.float16
        ).reshape(-1, h, w)

    async def embed_image[DType: np.generic](
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        output_dtype: type[DType] = np.float32,  # type: ignore[assignment]
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        **headers: str,
    ) -> NDArray[DType]:
        """Compute an embedding vector for an image using the specified model.

        Args:
            model: The name of the model to use for embedding.
            image: The image to embed. It must be uint8 RGB image.
            output_dtype: Output numpy dtype for embeddings (e.g. np.float16, np.float32).
            timeout: Optional timeout for the request.
            **headers: Additional x- headers. Keyword underscores are converted
                to hyphens and prefixed with 'x-', e.g. pool_tokens="false"
                becomes x-pool-tokens: false.

        Returns:
            NDArray[DType]: The embedding array reshaped according to
                the `x-output-shape` response header.
        """
        compressed_data = lz4.frame.compress(image.tobytes())
        request_headers = {"x-output-dtype": np.dtype(output_dtype).name}
        request_headers.update(
            {f"x-{k.replace('_', '-')}": v for k, v in headers.items()}
        )

        response = await self._post(
            model,
            data=compressed_data,
            headers=request_headers,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = lz4.frame.decompress(response.content)
        embedding = np.frombuffer(payload, dtype=output_dtype)

        response_shape = response.headers["x-output-shape"]
        embedding = embedding.reshape(eval(response_shape))

        return cast("NDArray[DType]", embedding)

    async def schedule(
        self,
        model: str,
        image: Image | NDArray[np.uint8],
        *,
        output_dtype: type[np.generic] | None = None,
        requested_timeout_seconds: int | None = None,
        idempotency_key: str | None = None,
        timeout: TimeoutTypes | UseClientDefault = USE_CLIENT_DEFAULT,
        **headers: str,
    ) -> ScheduledExecution:
        """Queue one tile for the model instead of waiting on the connection.

        The tile must be what the model's raw endpoint takes: a uint8 RGB image
        of its `tile_size`. Collect the outcome with `executions.wait` and
        `executions.result`; the response is kept for a limited time.

        Args:
            model: Model ID.
            image: The tile; must be a uint8 RGB image of the model's tile size.
            output_dtype: Output dtype for models that support `x-output-dtype`.
            requested_timeout_seconds: Optional server-side deadline.
            idempotency_key: Recovery key; generated if omitted. Reuse the same
                key after a client restart to recover an ambiguous submission.
            timeout: HTTP timeout for each submission attempt.
            **headers: Additional x- headers, e.g. pool_tokens="false".

        Returns:
            The admitted execution, replay flag, and idempotency key.
        """
        key = idempotency_key or str(uuid4())
        request = schedule_request(
            model,
            image,
            idempotency_key=key,
            output_dtype=output_dtype,
            requested_timeout_seconds=requested_timeout_seconds,
            headers=headers,
        )
        for attempt in range(SCHEDULE_ATTEMPTS):
            try:
                response = await self._post(
                    request.url,
                    content=request.content,
                    headers=request.headers,
                    params=request.params,
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
            return ScheduledExecution.from_dict(response.json(), idempotency_key=key)
        raise AssertionError("schedule retry loop did not return or raise")
