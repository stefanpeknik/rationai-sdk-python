# Executions

Execution polling exposes only `queued`, `running`, `succeeded`, `failed`, and
`cancelled`. A normalized result is available after success; failures expose a
safe structured error. `lookup` and `wait` fetch up to 1000 executions per
request. A scheduled tile's model response is downloaded with `result`, which
returns an `InferenceResponse`.

## `rationai.resources.executions.Executions`

::: rationai.resources.executions.Executions

## `rationai.resources.executions.AsyncExecutions`

::: rationai.resources.executions.AsyncExecutions
