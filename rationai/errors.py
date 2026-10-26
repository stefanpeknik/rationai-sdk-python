from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import httpx


class APIErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNKNOWN_BUILDER = "unknown_builder"
    UNKNOWN_MODEL = "unknown_model"
    INCOMPATIBLE_MODEL = "incompatible_model"
    SCHEDULE_CONFLICT = "schedule_conflict"
    SCHEDULING_UNAVAILABLE = "scheduling_unavailable"
    UNKNOWN_EXECUTION = "unknown_execution"
    BUILDER_BACKPRESSURE = "builder_backpressure"
    BUILDER_UNAVAILABLE = "builder_unavailable"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RESULT_UNAVAILABLE = "result_unavailable"


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


class InvalidRequestError(RationAIAPIError):
    pass


class UnknownBuilderError(RationAIAPIError):
    pass


class UnknownModelError(RationAIAPIError):
    pass


class IncompatibleModelError(RationAIAPIError):
    pass


class ScheduleConflictError(RationAIAPIError):
    pass


class SchedulingUnavailableError(RationAIAPIError):
    pass


class UnknownExecutionError(RationAIAPIError):
    pass


class BuilderBackpressureError(RationAIAPIError):
    pass


class BuilderUnavailableError(RationAIAPIError):
    pass


class PayloadTooLargeError(RationAIAPIError):
    pass


class ResultUnavailableError(RationAIAPIError):
    """The execution has no stored model response (yet, or ever)."""


_ERROR_TYPES: dict[str, type[RationAIAPIError]] = {
    APIErrorCode.INVALID_REQUEST: InvalidRequestError,
    APIErrorCode.UNKNOWN_BUILDER: UnknownBuilderError,
    APIErrorCode.UNKNOWN_MODEL: UnknownModelError,
    APIErrorCode.INCOMPATIBLE_MODEL: IncompatibleModelError,
    APIErrorCode.SCHEDULE_CONFLICT: ScheduleConflictError,
    APIErrorCode.SCHEDULING_UNAVAILABLE: SchedulingUnavailableError,
    APIErrorCode.UNKNOWN_EXECUTION: UnknownExecutionError,
    APIErrorCode.BUILDER_BACKPRESSURE: BuilderBackpressureError,
    APIErrorCode.BUILDER_UNAVAILABLE: BuilderUnavailableError,
    APIErrorCode.PAYLOAD_TOO_LARGE: PayloadTooLargeError,
    APIErrorCode.RESULT_UNAVAILABLE: ResultUnavailableError,
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
