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

## `rationai.resources.workflows.Workflows`

::: rationai.resources.workflows.Workflows

## `rationai.resources.workflows.AsyncWorkflows`

::: rationai.resources.workflows.AsyncWorkflows
