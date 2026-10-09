from __future__ import annotations

import json
import unittest
from datetime import datetime

import httpx
import lz4.frame
import numpy as np

import rationai
from rationai.errors import (
    InvalidWorkflowRequestError,
    ScheduleConflictError,
    SchedulingUnavailableError,
    UnknownExecutionError,
    UnknownWorkflowError,
)
from rationai.types import (
    ExecutionStatus,
    InlineOutput,
    InputAccess,
    SharedArtifact,
    SharedPathInput,
    WorkflowInputKind,
    WorkflowOutputKind,
    WorkflowRequest,
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


def workflow_request() -> WorkflowRequest:
    return WorkflowRequest(
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
        "workflow_id": "prostate-heatmap",
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


class WorkflowClientTests(unittest.TestCase):
    def test_run_and_schedule_accept_inputs_without_configuration(self) -> None:
        bodies: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            if request.url.path.endswith("/schedule"):
                return httpx.Response(
                    202, json={"execution": execution_body(), "created": True}
                )
            return httpx.Response(200, json=RESULT)

        request = WorkflowRequest(
            inputs={
                "slide": SharedPathInput("slide.svs"),
                "region_of_interest": SharedPathInput("roi.tiff"),
            },
            input_access=InputAccess("shared_storage", {"root": "/shared"}),
        )
        with rationai.Client(
            base_url="http://model-service", transport=httpx.MockTransport(handler)
        ) as client:
            client.workflows.run("prostate-heatmap", request)
            client.workflows.schedule(
                "prostate-heatmap", request, idempotency_key="defaults-1"
            )
        self.assertNotIn("resolved_configuration", bodies[0])
        self.assertNotIn("output_path", bodies[0])
        self.assertNotIn("resolved_configuration", bodies[1]["body"])
        self.assertNotIn("output_path", bodies[1]["body"])

    def test_legacy_model_url_configures_the_shared_origin(self) -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            if request.url.path.endswith("/run"):
                return httpx.Response(200, json=RESULT)
            return httpx.Response(200, json=0.75)

        with rationai.Client(
            models_base_url="https://example.invalid/model-service/",
            transport=httpx.MockTransport(handler),
        ) as client:
            client.models.classify_image(
                "classifier", np.zeros((2, 2, 3), dtype=np.uint8)
            )
            client.workflows.run("prostate-heatmap", workflow_request())

        self.assertEqual(
            paths,
            [
                "/model-service/classifier",
                "/model-service/v1/workflows/prostate-heatmap/run",
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

    def test_model_workflow_and_execution_calls_keep_url_prefix(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path.endswith("/run"):
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
            client.workflows.run("prostate-heatmap", workflow_request())
            scheduled = client.workflows.schedule(
                "prostate-heatmap", workflow_request()
            )
            client.executions.get(scheduled.execution.id)
            client.executions.cancel(scheduled.execution.id)

        self.assertEqual(
            requests,
            [
                ("POST", "/model-service/classifier"),
                ("POST", "/model-service/v1/workflows/prostate-heatmap/run"),
                ("POST", "/model-service/v1/workflows/prostate-heatmap/schedule"),
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
        self.assertEqual(captured[0].url, "http://model-service/classifier")
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
            result = client.workflows.run(
                "prostate-heatmap",
                workflow_request(),
                workflow_timeout_seconds=30,
            )

        self.assertEqual(captured[0].url.path, "/v1/workflows/prostate-heatmap/run")
        body = json.loads(captured[0].content)
        self.assertEqual(body["inputs"]["slide"]["kind"], "shared_path")
        self.assertEqual(
            workflow_request().inputs["slide"].kind, WorkflowInputKind.SHARED_PATH
        )
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
        self.assertEqual(result.outputs["summary"].kind, WorkflowOutputKind.INLINE)

    def test_generated_idempotency_key_is_reused_for_schedule_retry(self) -> None:
        payloads: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payloads.append(json.loads(request.content))
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
            scheduled = client.workflows.schedule(
                "prostate-heatmap",
                workflow_request(),
                requested_timeout_seconds=600,
            )

        self.assertEqual(len(payloads), 2)
        self.assertEqual(payloads[0], payloads[1])
        key = payloads[0]["idempotency_key"]
        self.assertIsInstance(key, str)
        self.assertEqual(scheduled.idempotency_key, key)
        self.assertTrue(scheduled.created)
        self.assertEqual(scheduled.execution.status, ExecutionStatus.QUEUED)

    def test_caller_idempotency_key_and_replay_are_exposed(self) -> None:
        captured: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"execution": execution_body(), "created": False},
            )

        with rationai.Client(
            base_url="http://model-service",
            transport=httpx.MockTransport(handler),
        ) as client:
            scheduled = client.workflows.schedule(
                "prostate-heatmap",
                workflow_request(),
                idempotency_key="recoverable-key",
            )

        self.assertEqual(captured[0]["idempotency_key"], "recoverable-key")
        self.assertEqual(scheduled.idempotency_key, "recoverable-key")
        self.assertFalse(scheduled.created)

    def test_public_error_responses_have_distinct_exception_types(self) -> None:
        cases = (
            ("invalid_schedule_request", 422, InvalidWorkflowRequestError),
            ("unknown_workflow", 404, UnknownWorkflowError),
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
                client.workflows.schedule(
                    "prostate-heatmap",
                    workflow_request(),
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
            if request.url.path.endswith("/run"):
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
            synchronous = client.workflows.run(
                "prostate-heatmap",
                workflow_request(),
            )
            scheduled = client.workflows.schedule(
                "prostate-heatmap",
                workflow_request(),
            )
            completed = client.executions.get(scheduled.execution.id)

        self.assertEqual(completed.result, synchronous)


class AsyncWorkflowClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_model_url_keeps_prefix_across_async_resources(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path.endswith("/run"):
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
            await client.workflows.run("prostate-heatmap", workflow_request())
            scheduled = await client.workflows.schedule(
                "prostate-heatmap", workflow_request()
            )
            await client.executions.get(scheduled.execution.id)
            await client.executions.cancel(scheduled.execution.id)

        self.assertEqual(
            requests,
            [
                ("POST", "/model-service/classifier"),
                ("POST", "/model-service/v1/workflows/prostate-heatmap/run"),
                ("POST", "/model-service/v1/workflows/prostate-heatmap/schedule"),
                ("GET", "/model-service/v1/executions/execution-1"),
                ("POST", "/model-service/v1/executions/execution-1/cancel"),
            ],
        )

    async def test_async_resources_use_the_same_typed_contract(self) -> None:
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append((request.method, request.url.path))
            if request.url.path.endswith("/run"):
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
            result = await client.workflows.run(
                "prostate-heatmap",
                workflow_request(),
            )
            scheduled = await client.workflows.schedule(
                "prostate-heatmap",
                workflow_request(),
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
                ("POST", "/v1/workflows/prostate-heatmap/run"),
                ("POST", "/v1/workflows/prostate-heatmap/schedule"),
                ("GET", "/v1/executions/execution-1"),
                ("POST", "/v1/executions/execution-1/cancel"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
