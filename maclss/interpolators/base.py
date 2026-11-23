"""Interpolator protocol — swap backends without changing the video pipeline."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameInterpolator(Protocol):
    """Produce an intermediate frame between two BGR uint8 frames.

    Implementations must return a frame with the same shape and dtype as
    ``frame_a`` / ``frame_b``. ``t`` is the blend position in ``(0, 1)``;
    for 2× interpolation the pipeline passes ``t=0.5``.
    """

    name: str

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        t: float = 0.5,
    ) -> np.ndarray:
        """Return a synthetic frame at temporal position ``t`` between A and B."""
        ...
