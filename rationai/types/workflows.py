from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal


type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowInputKind(StrEnum):
    INLINE = "inline"
    SHARED_PATH = "shared_path"
    PROVIDER = "provider"


class WorkflowOutputKind(StrEnum):
    INLINE = "inline"
    SHARED_ARTIFACT = "shared_artifact"


@dataclass(frozen=True, slots=True)
class InlineInput:
    value: JsonValue
    kind: Literal[WorkflowInputKind.INLINE] = WorkflowInputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    path: str
    kind: Literal[WorkflowInputKind.SHARED_PATH] = WorkflowInputKind.SHARED_PATH


@dataclass(frozen=True, slots=True)
class ProviderInput:
    name: str
    kind: Literal[WorkflowInputKind.PROVIDER] = WorkflowInputKind.PROVIDER


type WorkflowInput = InlineInput | SharedPathInput | ProviderInput


@dataclass(frozen=True, slots=True)
class InputAccess:
    kind: str
    state: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class WorkflowRequest:
    inputs: dict[str, WorkflowInput]
    resolved_configuration: dict[str, JsonValue]
    input_access: InputAccess

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs": {name: asdict(value) for name, value in self.inputs.items()},
            "resolved_configuration": self.resolved_configuration,
            "input_access": asdict(self.input_access),
        }


@dataclass(frozen=True, slots=True)
class InlineOutput:
    value: JsonValue
    kind: Literal[WorkflowOutputKind.INLINE] = WorkflowOutputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    path: str
    media_type: str | None = None
    kind: Literal[WorkflowOutputKind.SHARED_ARTIFACT] = (
        WorkflowOutputKind.SHARED_ARTIFACT
    )


type WorkflowOutput = InlineOutput | SharedArtifact


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    outputs: dict[str, WorkflowOutput]

    @classmethod
    def from_dict(cls, value: object) -> WorkflowResult:
        mapping = _mapping(value, "workflow result")
        outputs = _mapping(mapping.get("outputs"), "workflow outputs")
        return cls({name: _output(output) for name, output in outputs.items()})


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Execution:
    id: str
    workflow_id: str
    status: ExecutionStatus
    result: WorkflowResult | None = None
    error: ExecutionError | None = None

    @classmethod
    def from_dict(cls, value: object) -> Execution:
        mapping = _mapping(value, "execution")
        try:
            status = ExecutionStatus(_string(mapping, "status"))
        except ValueError as error:
            raise ValueError("execution has an unknown status") from error
        raw_result = mapping.get("result")
        raw_error = mapping.get("error")
        return cls(
            id=_string(mapping, "id"),
            workflow_id=_string(mapping, "workflow_id"),
            status=status,
            result=WorkflowResult.from_dict(raw_result)
            if raw_result is not None
            else None,
            error=_execution_error(raw_error) if raw_error is not None else None,
        )


@dataclass(frozen=True, slots=True)
class ScheduledExecution:
    execution: Execution
    created: bool
    idempotency_key: str

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        idempotency_key: str,
    ) -> ScheduledExecution:
        mapping = _mapping(value, "schedule response")
        created = mapping.get("created")
        if not isinstance(created, bool):
            raise TypeError("schedule response created must be a boolean")
        return cls(
            execution=Execution.from_dict(mapping.get("execution")),
            created=created,
            idempotency_key=idempotency_key,
        )


def _output(value: object) -> WorkflowOutput:
    mapping = _mapping(value, "workflow output")
    kind = mapping.get("kind")
    if kind == WorkflowOutputKind.INLINE:
        inline_value = mapping.get("value")
        _ensure_json(inline_value, "inline output")
        return InlineOutput(inline_value)
    if kind == WorkflowOutputKind.SHARED_ARTIFACT:
        media_type = mapping.get("media_type")
        if media_type is not None and not isinstance(media_type, str):
            raise ValueError("shared artifact media_type must be a string or null")
        return SharedArtifact(_string(mapping, "path"), media_type)
    raise ValueError("workflow output has an unknown kind")


def _execution_error(value: object) -> ExecutionError:
    mapping = _mapping(value, "execution error")
    return ExecutionError(_string(mapping, "code"), _string(mapping, "message"))


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _string(mapping: dict[str, Any], name: str) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ensure_json(value: object, name: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list):
        for item in value:
            _ensure_json(item, name)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _ensure_json(item, name)
        return
    raise ValueError(f"{name} must contain only JSON-compatible values")
