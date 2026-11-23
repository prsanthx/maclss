"""Video read → interpolate → write pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2

from maclss.interpolators.base import FrameInterpolator


ProgressCallback = Callable[[int, int], None]


def interpolate_video(
    input_path: Path | str,
    output_path: Path | str,
    interpolator: FrameInterpolator,
    factor: int = 2,
    progress: ProgressCallback | None = None,
) -> dict:
    """Insert ``factor - 1`` interpolated frames between each consecutive pair.

    For the v1 default ``factor=2``, one mid-frame is inserted between every
    pair, roughly doubling frame count and output FPS.

    Returns a small stats dict (input/output FPS, frame counts, backend name).
    """
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    try:
        input_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if input_fps <= 0:
            input_fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        output_fps = input_fps * factor
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = _pick_fourcc(output_path)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                f"Could not open video writer for {output_path}. "
                "Try an .mp4 or .avi extension."
            )

        try:
            ok, prev = cap.read()
            if not ok or prev is None:
                raise RuntimeError(f"Video has no frames: {input_path}")

            frames_in = 1
            frames_out = 0
            writer.write(prev)
            frames_out += 1

            while True:
                ok, curr = cap.read()
                if not ok or curr is None:
                    break
                frames_in += 1

                for step in range(1, factor):
                    t = step / factor
                    mid = interpolator.interpolate(prev, curr, t=t)
                    writer.write(mid)
                    frames_out += 1

                writer.write(curr)
                frames_out += 1
                prev = curr

                if progress is not None and reported_frames > 0:
                    progress(frames_in, reported_frames)
        finally:
            writer.release()
    finally:
        cap.release()

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "backend": getattr(interpolator, "name", type(interpolator).__name__),
        "factor": factor,
        "input_fps": input_fps,
        "output_fps": output_fps,
        "frames_in": frames_in,
        "frames_out": frames_out,
        "width": width,
        "height": height,
    }


def _pick_fourcc(path: Path) -> int:
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        # mp4v is widely available via OpenCV builds; ffmpeg can remux later.
        return cv2.VideoWriter_fourcc(*"mp4v")
    if suffix == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    # Default to mp4v for unknown containers.
    return cv2.VideoWriter_fourcc(*"mp4v")
