"""Smoke tests: interpolated output roughly doubles frames and FPS."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from maclss.demo import generate_synthetic_clip
from maclss.interpolators import OpticalFlowInterpolator, get_interpolator
from maclss.pipeline import interpolate_video


@pytest.fixture()
def synthetic_clip(tmp_path: Path) -> Path:
    path = tmp_path / "in_30fps.mp4"
    generate_synthetic_clip(path, fps=30.0, duration_s=0.5, width=160, height=120)
    return path


def test_get_default_interpolator() -> None:
    interp = get_interpolator("optical-flow")
    assert isinstance(interp, OpticalFlowInterpolator)


def test_optical_flow_produces_mid_frame(synthetic_clip: Path) -> None:
    cap = cv2.VideoCapture(str(synthetic_clip))
    ok_a, a = cap.read()
    ok_b, b = cap.read()
    cap.release()
    assert ok_a and ok_b

    mid = OpticalFlowInterpolator().interpolate(a, b, t=0.5)
    assert mid.shape == a.shape
    assert mid.dtype == a.dtype
    # Mid frame should not be an exact copy of either neighbor.
    assert not (mid == a).all()
    assert not (mid == b).all()


def test_pipeline_roughly_doubles_frames_and_fps(
    synthetic_clip: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out_60fps.mp4"
    stats = interpolate_video(
        synthetic_clip,
        out,
        interpolator=OpticalFlowInterpolator(),
        factor=2,
    )

    assert stats["frames_in"] >= 2
    # N input frames → N + (N-1) = 2N-1 output frames for factor=2.
    expected_out = 2 * stats["frames_in"] - 1
    assert stats["frames_out"] == expected_out
    assert stats["output_fps"] == pytest.approx(stats["input_fps"] * 2, rel=1e-3)

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    out_fps = float(cap.get(cv2.CAP_PROP_FPS))
    counted = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        counted += 1
    cap.release()

    assert counted == expected_out
    assert out_fps == pytest.approx(60.0, rel=0.05)
