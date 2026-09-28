# Workflows

`client.workflows.run` waits for a workflow result on the request it opens.
`client.workflows.schedule` returns a durable execution that can be recovered
through `client.executions`.

Both operations use the client's `base_url`. Scheduling generates an
idempotency key when none is supplied and reuses it for transport or temporary
availability retries. Pass an explicit key when a submission must be recovered
after restarting the client process.

Synchronous runs have no separate HTTP timeout by default; their absolute
workflow deadline is authoritative. Pass `timeout` explicitly when the client
transport should stop waiting sooner.

The example below requires the slide and ROI files to be visible to the
model-service under `/shared`. Paths are resolved on the server, not on the
client machine.

```python
from rationai import Client
from rationai.types import InputAccess, SharedPathInput, WorkflowRequest

request = WorkflowRequest(
    inputs={
        "slide": SharedPathInput("slides/input.svs"),
        "region_of_interest": SharedPathInput("masks/roi.tiff"),
    },
    resolved_configuration={
        "output_path": "outputs/heatmap.tiff",
        "stride_fraction": 0.5,
        "output_tile_height": 256,
        "output_tile_width": 256,
    },
    input_access=InputAccess(kind="shared_storage", state={"root": "/shared"}),
)

mode = "scheduled"  # Change to "immediate" to wait for the result instead.
with Client(base_url="http://localhost:8080") as client:
    if mode == "immediate":
        print(client.workflows.run("prostate-heatmap", request).outputs)
    else:
        scheduled = client.workflows.schedule(
            "prostate-heatmap", request, idempotency_key="heatmap-001"
        )
        current = client.executions.get(scheduled.execution.id)
        print(current.status)
```

Use a distinct idempotency key for each new scheduled request. Reuse the same
key only to recover or replay that request. A scheduled result is available
through `client.executions.get(...)` after its status becomes `succeeded`.

## `rationai.resources.workflows.Workflows`

::: rationai.resources.workflows.Workflows

## `rationai.resources.workflows.AsyncWorkflows`

::: rationai.resources.workflows.AsyncWorkflows
