"""Frame interpolators.

Default backend is OpenCV Farneback optical flow. A RIFE-style neural
backend can be plugged in later via the same ``FrameInterpolator`` protocol
(see ``base.py`` and ``rife.py`` stub).
"""

from __future__ import annotations

from maclss.interpolators.base import FrameInterpolator
from maclss.interpolators.optical_flow import OpticalFlowInterpolator


def get_interpolator(name: str = "optical-flow") -> FrameInterpolator:
    """Resolve an interpolator by short name.

    Parameters
    ----------
    name:
        ``optical-flow`` (default) or ``rife`` (requires optional deps).
    """
    key = name.strip().lower().replace("_", "-")
    if key in {"optical-flow", "flow", "opencv", "farneback"}:
        return OpticalFlowInterpolator()
    if key == "rife":
        from maclss.interpolators.rife import RifeInterpolator

        return RifeInterpolator()
    raise ValueError(
        f"Unknown interpolator {name!r}. "
        "Available: optical-flow, rife (optional)."
    )


__all__ = [
    "FrameInterpolator",
    "OpticalFlowInterpolator",
    "get_interpolator",
]
