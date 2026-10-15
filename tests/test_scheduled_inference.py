from __future__ import annotations

import json
import unittest

import httpx
import lz4.frame
import numpy as np

import rationai
from rationai.errors import PayloadTooLargeError, ResultUnavailableError
from rationai.types import ExecutionStatus, InferenceResult, Operation
from tests.test_builders import api_error


def inference_view(
    execution_id: str = "tile-1", status: str = "queued", **extra: object
) -> dict[str, object]:
    return {
        "id": execution_id,
        "operation": "inference",
        "builder_id": None,
        "model_id": "virchow2",
        "status": status,
        "result": None,
        "error": None,
        **extra,
    }


def client(handler: httpx.MockTransport) -> rationai.Client:
    return rationai.Client(base_url="http://model-service", transport=handler)


class ScheduledTileTests(unittest.TestCase):
    def test_tile_is_compressed_and_retried_with_the_same_key(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return api_error("scheduling_unavailable", 503)
            return httpx.Response(
                202, json={"execution": inference_view(), "created": True}
            )

        tile = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        with client(httpx.MockTransport(handler)) as sdk:
            scheduled = sdk.models.schedule(
                "virchow2",
                tile,
                output_dtype=np.float16,
                requested_timeout_seconds=600,
                pool_tokens="false",
            )

        first, second = requests
        self.assertEqual(
            str(first.url),
            "http://model-service/v1/models/virchow2/schedule?timeout_seconds=600",
        )
        self.assertEqual(lz4.frame.decompress(first.content), tile.tobytes())
        self.assertEqual(first.headers["x-output-dtype"], "float16")
        self.assertEqual(first.headers["x-pool-tokens"], "false")
        self.assertEqual(
            first.headers["idempotency-key"], second.headers["idempotency-key"]
        )
        self.assertEqual(scheduled.idempotency_key, first.headers["idempotency-key"])
        execution = scheduled.execution
        self.assertEqual(execution.operation, Operation.INFERENCE)
        self.assertIsNone(execution.builder_id)
        self.assertEqual(execution.status, ExecutionStatus.QUEUED)

    def test_oversized_tile_has_its_own_error(self) -> None:
        with (
            client(
                httpx.MockTransport(lambda _r: api_error("payload_too_large", 413))
            ) as sdk,
            self.assertRaises(PayloadTooLargeError),
        ):
            sdk.models.schedule("virchow2", np.zeros((2, 2, 3), dtype=np.uint8))

    def test_lookup_batches_ids_and_wait_returns_when_all_finished(self) -> None:
        ids = [f"tile-{index}" for index in range(1500)]
        lookups: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested = json.loads(request.content)["ids"]
            lookups.append(requested)
            status = "succeeded" if len(lookups) > 2 else "running"
            return httpx.Response(
                200,
                json=[inference_view(value, status) for value in requested],
            )

        with client(httpx.MockTransport(handler)) as sdk:
            finished = sdk.executions.wait(ids, poll_interval=0)

        self.assertEqual([len(batch) for batch in lookups], [1000, 500, 1000, 500])
        self.assertEqual([execution.id for execution in finished], ids)
        self.assertTrue(all(execution.finished for execution in finished))

    def test_result_metadata_and_downloaded_response_decode(self) -> None:
        embedding = np.arange(6, dtype=np.float32).reshape(2, 3)
        view = inference_view(
            status="succeeded",
            result={
                "media_type": "application/octet-stream",
                "headers": {"x-output-shape": "(2, 3)"},
                "size": 40,
            },
        )

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/result"):
                return httpx.Response(
                    200,
                    content=lz4.frame.compress(embedding.tobytes()),
                    headers={
                        "content-type": "application/octet-stream",
                        "x-output-shape": "(2, 3)",
                    },
                )
            return httpx.Response(200, json=view)

        with client(httpx.MockTransport(handler)) as sdk:
            execution = sdk.executions.get("tile-1")
            response = sdk.executions.result("tile-1")

        self.assertEqual(
            execution.result,
            InferenceResult(
                "application/octet-stream", {"x-output-shape": "(2, 3)"}, 40
            ),
        )
        np.testing.assert_array_equal(response.array(np.float32), embedding)
        self.assertEqual(response.media_type, "application/octet-stream")

    def test_json_response_and_missing_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "missing" in request.url.path:
                return api_error("result_unavailable", 409)
            return httpx.Response(200, json=0.75)

        with client(httpx.MockTransport(handler)) as sdk:
            self.assertEqual(sdk.executions.result("tile-1").json(), 0.75)
            with self.assertRaises(ResultUnavailableError):
                sdk.executions.result("missing")


class AsyncScheduledTileTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_schedule_wait_and_result(self) -> None:
        polls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202, json={"execution": inference_view(), "created": True}
                )
            if request.url.path.endswith("/lookup"):
                polls += 1
                status = "succeeded" if polls > 1 else "queued"
                return httpx.Response(200, json=[inference_view(status=status)])
            return httpx.Response(200, json=0.25)

        async with rationai.AsyncClient(
            base_url="http://model-service", transport=httpx.MockTransport(handler)
        ) as sdk:
            scheduled = await sdk.models.schedule(
                "classifier", np.zeros((2, 2, 3), dtype=np.uint8)
            )
            [finished] = await sdk.executions.wait(
                [scheduled.execution.id], poll_interval=0
            )
            response = await sdk.executions.result(finished.id)

        self.assertEqual(polls, 2)
        self.assertEqual(response.json(), 0.25)


if __name__ == "__main__":
    unittest.main()
