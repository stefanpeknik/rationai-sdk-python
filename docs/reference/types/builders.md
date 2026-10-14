# Builder types

Builder inputs and outputs use explicit tagged dataclasses. In particular, an
inline string remains an inline value and is never guessed to be an artifact
path.

`BuilderRequest` contains inputs, input access, and an optional `output_path`.
Processing settings belong to model-service and are selected when the builder
starts, including each scheduled attempt. Omitting `output_path` uses the
builder's default destination. The old `resolved_configuration` argument is
no longer accepted; update callers and model-service together.

::: rationai.types.builders
