"""OpenCV Farneback optical-flow frame interpolator.

Computes dense forward/backward flow, warps both frames toward the midpoint,
and blends them. This is real motion-compensated interpolation — not frame
duplication or a simple alpha blend.
"""

from __future__ import annotations

import cv2
import numpy as np


class OpticalFlowInterpolator:
    """Farneback-based interpolator suitable for CPU-only environments."""

    name = "optical-flow"

    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        self._flow_kwargs = {
            "pyr_scale": pyr_scale,
            "levels": levels,
            "winsize": winsize,
            "iterations": iterations,
            "poly_n": poly_n,
            "poly_sigma": poly_sigma,
            "flags": 0,
        }

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        t: float = 0.5,
    ) -> np.ndarray:
        if frame_a.shape != frame_b.shape:
            raise ValueError(
                f"Frame shape mismatch: {frame_a.shape} vs {frame_b.shape}"
            )
        if not (0.0 < t < 1.0):
            raise ValueError(f"t must be in (0, 1), got {t}")

        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        flow_ab = cv2.calcOpticalFlowFarneback(
            gray_a, gray_b, None, **self._flow_kwargs
        )
        flow_ba = cv2.calcOpticalFlowFarneback(
            gray_b, gray_a, None, **self._flow_kwargs
        )

        warped_a = self._warp(frame_a, flow_ab * t)
        warped_b = self._warp(frame_b, flow_ba * (1.0 - t))

        blended = cv2.addWeighted(warped_a, 1.0 - t, warped_b, t, 0.0)
        return blended

    @staticmethod
    def _warp(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp ``frame`` along ``flow`` (dx, dy per pixel)."""
        h, w = flow.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        return cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
