from __future__ import annotations

import json
import unittest
from datetime import datetime

import httpx
import lz4.frame
import numpy as np

import rationai
from rationai.errors import (
    InvalidRequestError,
    ScheduleConflictError,
    SchedulingUnavailableError,
    UnknownBuilderError,
    UnknownExecutionError,
)
from rationai.types import (
    BuilderRequest,
    ExecutionStatus,
    InlineOutput,
    InputAccess,
    InputKind,
    OutputKind,
    SharedArtifact,
    SharedPathInput,
)


RESULT = {
    "outputs": {
        "heatmap": {
            "kind": "shared_artifact",
            "path": "/shared/outputs/heatmap.tiff",
            "media_type": "image/tiff",
        },
        "summary": {"kind": "inline", "value": {"tiles": 42}},
    }
}


def builder_request() -> BuilderRequest:
    return BuilderRequest(
        model_id="prostate-classifier-1",
        inputs={
            "slide": SharedPathInput("slides/input.svs"),
            "region_of_interest": SharedPathInput("masks/roi.tiff"),
        },
        output_path="outputs/heatmap.tiff",
        input_access=InputAccess(
            kind="shared_storage",
            state={"root": "/shared"},
        ),
    )


def execution_body(
    *,
    status: str = "queued",
    result: object = None,
    error: object = None,
) -> dict[str, object]:
    return {
        "id": "execution-1",
        "builder_id": "heatmap",
        "model_id": "prostate-classifier-1",
        "status": status,
        "result": result,
        "error": error,
    }


def api_error(code: str, status_code: int) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "detail": {
                "code": code,
                "message": f"safe {code}",
                "retryable": code == "scheduling_unavailable",
            }
        },
        headers={"Retry-After": "0"},
    )


class BuilderClientTests(unittest.TestCase):
    def test_run_and_schedule_accept_inputs_without_configuration(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202, json={"execution": execution_body(), "created": True}
                )
            return httpx.Response(200, json=RESULT)

        request = BuilderRequest(
            model_id="prostate-classifier-1",
            inputs={
                "slide": SharedPathInput("slide.svs"),
                "region_of_interest": SharedPathInput("roi.tiff"),
            },
            input_access=InputAccess("shared_storage", {"root": "/shared"}),
        )
        with rationai.Client(
            base_url="http://model-service", transport=httpx.MockTransport(handler)
        ) as client:
            client.builders.run("heatmap", request)
            client.builders.schedule("heatmap", request, idempotency_key="defaults-1")
        self.assertNotIn("resolved_configuration", bodies[0])
        self.assertNotIn("output_path", bodies[0])
        self.assertNotIn("resolved_configuration", bodies[1])
        self.assertNotIn("output_path", bodies[1])

    def test_legacy_model_url_configures_the_shared_origin(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if "/v1/builders/" in request.url.path and request.url.path.endswith(
                "/run"
            ):
                return httpx.Response(200, json=RESULT)
            return httpx.Response(200, json=0.75)

        with rationai.Client(
            models_base_url="https://example.invalid/model-service/",
            transport=httpx.MockTransport(handler),
        ) as client:
            client.models.classify_image(
                "classifier", np.zeros((2, 2, 3), dtype=np.uint8)
            )
            client.builders.run("heatmap", builder_request())

        self.assertEqual(
            paths,
            [
                "/model-service/v1/models/classifier/run",
                "/model-service/v1/builders/heatmap/models/prostate-classifier-1/run",
            ],
        )

    def test_conflicting_model_service_urls_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rationai.Client(
                base_url="https://one.invalid/",
                models_base_url="https://two.invalid/",
            )
        with self.assertRaises(ValueError):
            rationai.AsyncClient(
                base_url="https://one.invalid/",
                models_base_url="https://two.invalid/",
            )

    def test_model_builder_and_execution_calls_keep_url_prefix(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if "/v1/builders/" in request.url.path and request.url.path.endswith(
                "/run"
            ):
                return httpx.Response(200, json=RESULT)
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202, json={"execution": execution_body(), "created": True}
                )
            if "/executions/" in request.url.path:
                return httpx.Response(200, json=execution_body())
            return httpx.Response(200, json=0.75)

        with rationai.Client(
            base_url="https://example.invalid/model-service/",
            transport=httpx.MockTransport(handler),
        ) as client:
            client.models.classify_image(
                "classifier", np.zeros((2, 2, 3), dtype=np.uint8)
            )
            client.builders.run("heatmap", builder_request())
            scheduled = client.builders.schedule("heatmap", builder_request())
            client.executions.get(scheduled.execution.id)
            client.executions.cancel(scheduled.execution.id)

        self.assertEqual(
            requests,
            [
                ("POST", "/model-service/v1/models/classifier/run"),
                (
                    "POST",
                    "/model-service/v1/builders/heatmap/models/prostate-classifier-1/run",
                ),
                (
                    "POST",
                    "/model-service/v1/builders/heatmap/models/prostate-classifier-1/schedule",
                ),
                ("GET", "/model-service/v1/executions/execution-1"),
                ("POST", "/model-service/v1/executions/execution-1/cancel"),
            ],
        )

    def test_existing_model_inference_uses_public_origin_and_compressed_body(
        self,
    ) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=0.75)

        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            result = client.models.classify_image("classifier", image)

        self.assertEqual(result, 0.75)
        self.assertEqual(
            captured[0].url, "http://model-service/v1/models/classifier/run"
        )
        self.assertEqual(lz4.frame.decompress(captured[0].content), image.tobytes())

    def test_run_serializes_tagged_inputs_and_parses_tagged_outputs(self) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=RESULT)

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            result = client.builders.run(
                "heatmap",
                builder_request(),
                run_timeout_seconds=30,
            )

        self.assertEqual(
            captured[0].url.path,
            "/v1/builders/heatmap/models/prostate-classifier-1/run",
        )
        body = json.loads(captured[0].content)
        self.assertEqual(body["inputs"]["slide"]["kind"], "shared_path")
        self.assertEqual(builder_request().inputs["slide"].kind, InputKind.SHARED_PATH)
        self.assertEqual(body["input_access"]["kind"], "shared_storage")
        self.assertEqual(body["output_path"], "outputs/heatmap.tiff")
        self.assertNotIn("resolved_configuration", body)
        self.assertIsNotNone(datetime.fromisoformat(body["deadline_at"]).tzinfo)
        self.assertEqual(
            result.outputs["heatmap"],
            SharedArtifact(
                "/shared/outputs/heatmap.tiff",
                "image/tiff",
            ),
        )
        self.assertEqual(result.outputs["summary"], InlineOutput({"tiles": 42}))
        self.assertEqual(result.outputs["summary"].kind, OutputKind.INLINE)

    def test_generated_idempotency_key_is_reused_for_schedule_retry(self) -> None:
        payloads: list[dict[str, object]] = []
        keys: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
            keys.append(request.headers["idempotency-key"])
            if len(payloads) == 1:
                return api_error("scheduling_unavailable", 503)
            return httpx.Response(
                202,
                json={"execution": execution_body(), "created": True},
            )

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            scheduled = client.builders.schedule(
                "heatmap",
                builder_request(),
                requested_timeout_seconds=600,
            )

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0], payloads[1])
        self.assertNotIn("idempotency_key", payloads[0])
        self.assertEqual(keys[0], keys[1])
        self.assertEqual(scheduled.idempotency_key, keys[0])
        self.assertTrue(scheduled.created)
        self.assertEqual(scheduled.execution.status, ExecutionStatus.QUEUED)

    def test_caller_idempotency_key_and_replay_are_exposed(self) -> None:
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.headers["idempotency-key"])
            return httpx.Response(
                200,
                json={"execution": execution_body(), "created": False},
            )

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            scheduled = client.builders.schedule(
                "heatmap",
                builder_request(),
                idempotency_key="recoverable-key",
            )

        self.assertEqual(captured, ["recoverable-key"])
        self.assertEqual(scheduled.idempotency_key, "recoverable-key")
        self.assertFalse(scheduled.created)

    def test_public_error_responses_have_distinct_exception_types(self) -> None:
        cases = (
            ("invalid_request", 422, InvalidRequestError),
            ("unknown_builder", 404, UnknownBuilderError),
            ("schedule_conflict", 409, ScheduleConflictError),
            ("scheduling_unavailable", 503, SchedulingUnavailableError),
        )
        for code, status_code, error_type in cases:
            with (
                self.subTest(code=code),
                rationai.Client(
                    base_url="http://model-service",
                    transport=httpx.MockTransport(
                        lambda _request, code=code, status_code=status_code: api_error(
                            code,
                            status_code,
                        )
                    ),
                ) as client,
                self.assertRaises(error_type),
            ):
                client.builders.schedule(
                    "heatmap",
                    builder_request(),
                    idempotency_key="request-1",
                )

    def test_executions_get_and_cancel_parse_only_public_state(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path.endswith("missing"):
                return api_error("unknown_execution", 404)
            return httpx.Response(
                200,
                json=execution_body(
                    status="succeeded",
                    result=RESULT,
                ),
            )

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            execution = client.executions.get("execution-1")
            cancelled = client.executions.cancel("execution-1")
            with self.assertRaises(UnknownExecutionError):
                client.executions.get("missing")

        self.assertEqual(execution.status, ExecutionStatus.SUCCEEDED)
        self.assertIsNotNone(execution.result)
        self.assertIsNotNone(cancelled.result)
        assert execution.result is not None
        assert cancelled.result is not None
        self.assertEqual(execution.result.outputs, cancelled.result.outputs)
        self.assertEqual(
            requests,
            [
                ("GET", "/v1/executions/execution-1"),
                ("POST", "/v1/executions/execution-1/cancel"),
                ("GET", "/v1/executions/missing"),
            ],
        )

    def test_run_and_scheduled_execution_have_equivalent_results(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if "/v1/builders/" in request.url.path and request.url.path.endswith(
                "/run"
            ):
                return httpx.Response(200, json=RESULT)
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202,
                    json={"execution": execution_body(), "created": True},
                )
            if request.url.path == "/v1/executions/execution-1":
                return httpx.Response(
                    200,
                    json=execution_body(status="succeeded", result=RESULT),
                )
            raise AssertionError(f"unexpected request {request.method} {request.url}")

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            synchronous = client.builders.run(
                "heatmap",
                builder_request(),
            )
            scheduled = client.builders.schedule(
                "heatmap",
                builder_request(),
            )
            completed = client.executions.get(scheduled.execution.id)

        self.assertEqual(completed.result, synchronous)


class AsyncBuilderClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_model_url_keeps_prefix_across_async_resources(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if "/v1/builders/" in request.url.path and request.url.path.endswith(
                "/run"
            ):
                return httpx.Response(200, json=RESULT)
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202, json={"execution": execution_body(), "created": True}
                )
            if "/executions/" in request.url.path:
                return httpx.Response(200, json=execution_body())
            return httpx.Response(200, json=0.75)

        async with rationai.AsyncClient(
            models_base_url="https://example.invalid/model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            await client.models.classify_image(
                "classifier", np.zeros((2, 2, 3), dtype=np.uint8)
            )
            await client.builders.run("heatmap", builder_request())
            scheduled = await client.builders.schedule("heatmap", builder_request())
            await client.executions.get(scheduled.execution.id)
            await client.executions.cancel(scheduled.execution.id)

        self.assertEqual(
            requests,
            [
                ("POST", "/model-service/v1/models/classifier/run"),
                (
                    "POST",
                    "/model-service/v1/builders/heatmap/models/prostate-classifier-1/run",
                ),
                (
                    "POST",
                    "/model-service/v1/builders/heatmap/models/prostate-classifier-1/schedule",
                ),
                ("GET", "/model-service/v1/executions/execution-1"),
                ("POST", "/model-service/v1/executions/execution-1/cancel"),
            ],
        )

    async def test_async_resources_use_the_same_typed_contract(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if "/v1/builders/" in request.url.path and request.url.path.endswith(
                "/run"
            ):
                return httpx.Response(200, json=RESULT)
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202,
                    json={"execution": execution_body(), "created": True},
                )
            if request.url.path.endswith("/cancel"):
                return httpx.Response(
                    200,
                    json=execution_body(status="cancelled"),
                )
            return httpx.Response(200, json=execution_body())

        async with rationai.AsyncClient(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            result = await client.builders.run(
                "heatmap",
                builder_request(),
            )
            scheduled = await client.builders.schedule(
                "heatmap",
                builder_request(),
                idempotency_key="async-request",
            )
            execution = await client.executions.get(scheduled.execution.id)
            cancelled = await client.executions.cancel(scheduled.execution.id)

        self.assertIsInstance(result.outputs["heatmap"], SharedArtifact)
        self.assertEqual(scheduled.idempotency_key, "async-request")
        self.assertEqual(execution.status, ExecutionStatus.QUEUED)
        self.assertEqual(cancelled.status, ExecutionStatus.CANCELLED)
        self.assertEqual(
            requests,
            [
                ("POST", "/v1/builders/heatmap/models/prostate-classifier-1/run"),
                ("POST", "/v1/builders/heatmap/models/prostate-classifier-1/schedule"),
                ("GET", "/v1/executions/execution-1"),
                ("POST", "/v1/executions/execution-1/cancel"),
            ],
        )


if __name__ == "__main__":
    unittest.main()


class CatalogTests(unittest.TestCase):
    def test_models_and_builders_are_listed(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(
                    200,
                    json=[
                        {
                            "id": "virchow2",
                            "contract": "tile-embedding",
                            "parameters": {"tile_size": 224, "mpp": 0.5},
                            "builders": ["tile-embeddings"],
                        }
                    ],
                )
            return httpx.Response(
                200,
                json=[
                    {
                        "id": "heatmap",
                        "accepts": "spatial-scores",
                        "inputs": {"slide": {"required": True, "media_type": None}},
                        "outputs": {"heatmap": {"kind": "shared_artifact"}},
                        "configuration": {"stride_fraction": 0.5},
                    }
                ],
            )

        with rationai.Client(
            base_url="http://model-service", transport=httpx.MockTransport(handler)
        ) as client:
            models = client.models.list()
            builders = client.builders.list()

        self.assertEqual(models[0].builders, ["tile-embeddings"])
        self.assertEqual(models[0].parameters["mpp"], 0.5)
        self.assertEqual(builders[0].accepts, "spatial-scores")

    def test_model_selection_errors_have_their_own_types(self) -> None:
        cases = {
            "unknown_model": rationai.UnknownModelError,
            "incompatible_model": rationai.IncompatibleModelError,
            "unknown_builder": rationai.UnknownBuilderError,
        }
        for code, error_type in cases.items():
            with (
                self.subTest(code=code),
                rationai.Client(
                    base_url="http://model-service",
                    transport=httpx.MockTransport(
                        lambda _request, code=code: api_error(code, 404)
                    ),
                ) as client,
                self.assertRaises(error_type),
            ):
                client.builders.run("heatmap", builder_request())
