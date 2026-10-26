# Inference types

`InferenceResponse` is the stored response of a scheduled tile. `.json()`
decodes a classifier's score; `.array(dtype)` decompresses an LZ4 array and
shapes it from the model's `x-output-shape` header.

::: rationai.types.inference
