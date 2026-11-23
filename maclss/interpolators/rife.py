"""RIFE-style neural interpolator stub.

Not wired in v1: installing torch + downloading model weights is heavy for
many environments. Structure the import so callers can select ``--backend rife``
once a concrete implementation lands.

Extension point for later:
- Drop in a real ``RifeInterpolator.interpolate`` that loads a RIFE / IFNet
  checkpoint and runs mid-frame inference.
- Keep the same signature as ``OpticalFlowInterpolator`` so the video
  pipeline does not change.
"""

from __future__ import annotations

import numpy as np


class RifeInterpolator:
    """Placeholder for a quality neural interpolator (RIFE / similar)."""

    name = "rife"

    def __init__(self) -> None:
        raise NotImplementedError(
            "RIFE backend is not bundled yet. Install optional extras "
            "(`pip install 'maclss[rife]'`) and implement model loading in "
            "maclss.interpolators.rife, or use --backend optical-flow."
        )

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        t: float = 0.5,
    ) -> np.ndarray:
        raise NotImplementedError("RIFE backend is not bundled yet.")
