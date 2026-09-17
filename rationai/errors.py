from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import httpx


class APIErrorCode(StrEnum):
    INVALID_SCHEDULE_REQUEST = "invalid_schedule_request"
    INVALID_WORKFLOW_INVOCATION = "invalid_workflow_invocation"
    UNKNOWN_WORKFLOW = "unknown_workflow"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    WORKFLOW_BACKPRESSURE = "workflow_backpressure"
    WORKFLOW_DISPATCH_UNAVAILABLE = "workflow_dispatch_unavailable"


class RationAIAPIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: APIErrorCode | str,
        status_code: int | None,
        retryable: bool,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable


class InvalidWorkflowRequestError(RationAIAPIError):
    pass


class UnknownWorkflowError(RationAIAPIError):
    pass


class ScheduleConflictError(RationAIAPIError):
    pass


class SchedulingUnavailableError(RationAIAPIError):
    pass


class UnknownExecutionError(RationAIAPIError):
    pass


class WorkflowBackpressureError(RationAIAPIError):
    pass


class WorkflowDispatchUnavailableError(RationAIAPIError):
    pass


_ERROR_TYPES: dict[str, type[RationAIAPIError]] = {
    APIErrorCode.INVALID_SCHEDULE_REQUEST: InvalidWorkflowRequestError,
    APIErrorCode.INVALID_WORKFLOW_INVOCATION: InvalidWorkflowRequestError,
    APIErrorCode.UNKNOWN_WORKFLOW: UnknownWorkflowError,
    APIErrorCode.SCHEDULE_CONFLICT: ScheduleConflictError,
    APIErrorCode.SCHEDULING_UNAVAILABLE: SchedulingUnavailableError,
    APIErrorCode.UNKNOWN_EXECUTION: UnknownExecutionError,
    APIErrorCode.WORKFLOW_BACKPRESSURE: WorkflowBackpressureError,
    APIErrorCode.WORKFLOW_DISPATCH_UNAVAILABLE: WorkflowDispatchUnavailableError,
}


def api_error(response: httpx.Response) -> RationAIAPIError | None:
    if response.status_code < 400:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    detail = body.get("detail")
    if not isinstance(detail, dict):
        return None
    code = detail.get("code")
    message = detail.get("message")
    retryable = detail.get("retryable")
    if (
        not isinstance(code, str)
        or not code
        or not isinstance(message, str)
        or not message
        or not isinstance(retryable, bool)
    ):
        return None
    error_type = _ERROR_TYPES.get(code, RationAIAPIError)
    try:
        typed_code: APIErrorCode | str = APIErrorCode(code)
    except ValueError:
        typed_code = code
    return error_type(
        message,
        code=typed_code,
        status_code=response.status_code,
        retryable=retryable,
    )


def raise_for_api_error(response: httpx.Response) -> None:
    error = api_error(response)
    if error is not None:
        raise error
    response.raise_for_status()
