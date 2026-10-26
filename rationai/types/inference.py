from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any

import lz4.frame
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class InferenceResponse:
    """The stored response of a scheduled tile, as the model returned it."""

    content: bytes
    media_type: str | None
    headers: dict[str, str]

    def json(self) -> Any:
        """Decode a JSON response, e.g. a classifier's score."""
        return json.loads(self.content)

    def array[DType: np.generic](
        self,
        dtype: type[DType],
        shape: tuple[int, ...] | None = None,
    ) -> NDArray[DType]:
        """Decode an LZ4-compressed array, e.g. an embedding or segmentation.

        The shape comes from `shape`, else from the model's `x-output-shape`
        header, else the array is flat.
        """
        values = np.frombuffer(lz4.frame.decompress(self.content), dtype=dtype)
        if shape is None and "x-output-shape" in self.headers:
            shape = tuple(ast.literal_eval(self.headers["x-output-shape"]))
        return values.reshape(shape) if shape is not None else values
