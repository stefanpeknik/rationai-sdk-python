"""Build `POST /v1/models/{model}/schedule` requests for one image tile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote

import lz4.frame
import numpy as np

from rationai.resources._scheduling import check_timeout


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class TileRequest:
    url: str
    content: bytes
    headers: dict[str, str]
    params: dict[str, int] | None


def schedule_request(
    model: str,
    image: Image | NDArray[np.uint8],
    *,
    idempotency_key: str,
    output_dtype: type[np.generic] | None,
    requested_timeout_seconds: int | None,
    headers: dict[str, str],
) -> TileRequest:
    check_timeout(requested_timeout_seconds)
    if not idempotency_key:
        raise ValueError("idempotency_key must be non-empty")
    request_headers = {"idempotency-key": idempotency_key}
    if output_dtype is not None:
        request_headers["x-output-dtype"] = np.dtype(output_dtype).name
    request_headers.update({f"x-{k.replace('_', '-')}": v for k, v in headers.items()})
    return TileRequest(
        url=f"v1/models/{quote(model, safe='')}/schedule",
        content=lz4.frame.compress(image.tobytes()),
        headers=request_headers,
        params=(
            {"timeout_seconds": requested_timeout_seconds}
            if requested_timeout_seconds is not None
            else None
        ),
    )
