from __future__ import annotations

import os
import time
import unittest
from uuid import uuid4

import rationai
from rationai.types import (
    BuilderRequest,
    ExecutionStatus,
    InputAccess,
    SharedPathInput,
)


class BuilderIntegrationTests(unittest.TestCase):
    def test_run_and_schedule_return_equivalent_results(self) -> None:
        required = {
            name: os.environ.get(name)
            for name in (
                "RATIONAI_MODEL_SERVICE_URL",
                "RATIONAI_BUILDER_SHARED_ROOT",
                "RATIONAI_BUILDER_SLIDE_PATH",
                "RATIONAI_BUILDER_ROI_PATH",
            )
        }
        missing = tuple(name for name, value in required.items() if not value)
        if missing:
            self.skipTest("builder integration environment is not configured")
        service_url = required["RATIONAI_MODEL_SERVICE_URL"]
        shared_root = required["RATIONAI_BUILDER_SHARED_ROOT"]
        slide_path = required["RATIONAI_BUILDER_SLIDE_PATH"]
        roi_path = required["RATIONAI_BUILDER_ROI_PATH"]
        assert service_url is not None
        assert shared_root is not None
        assert slide_path is not None
        assert roi_path is not None
        timeout_seconds = int(
            os.environ.get("RATIONAI_BUILDER_TIMEOUT_SECONDS", "3600")
        )
        request = BuilderRequest(
            model_id=os.environ.get("RATIONAI_TEST_MODEL", "prostate-classifier-1"),
            inputs={
                "slide": SharedPathInput(slide_path),
                "region_of_interest": SharedPathInput(roi_path),
            },
            output_path=f"outputs/sdk-integration-{uuid4()}.tiff",
            input_access=InputAccess(
                kind="shared_storage",
                state={"root": shared_root},
            ),
        )

        with rationai.Client(base_url=service_url) as client:
            synchronous = client.builders.run(
                "heatmap",
                request,
                run_timeout_seconds=timeout_seconds,
            )
            scheduled = client.builders.schedule(
                "heatmap",
                request,
                requested_timeout_seconds=timeout_seconds,
                idempotency_key=f"sdk-integration-{uuid4()}",
            )
            deadline = time.monotonic() + timeout_seconds
            execution = scheduled.execution
            while (
                execution.status in {ExecutionStatus.QUEUED, ExecutionStatus.RUNNING}
                and time.monotonic() < deadline
            ):
                time.sleep(2)
                execution = client.executions.get(execution.id)

        self.assertEqual(execution.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(execution.result, synchronous)


if __name__ == "__main__":
    unittest.main()
