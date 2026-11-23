"""Generate a short synthetic ~30 FPS clip for demos and smoke tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_synthetic_clip(
    output_path: Path | str,
    *,
    fps: float = 30.0,
    duration_s: float = 1.0,
    width: int = 320,
    height: int = 240,
) -> Path:
    """Write a moving-ball clip so interpolation has real motion to fill.

    Returns the output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_count = max(2, int(round(fps * duration_s)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {output_path}")

    try:
        radius = max(8, min(width, height) // 12)
        for i in range(frame_count):
            t = i / max(frame_count - 1, 1)
            # Bounce diagonally so optical flow has clear motion.
            cx = int(radius + t * (width - 2 * radius))
            cy = int(radius + (0.5 + 0.5 * np.sin(t * np.pi * 2)) * (height - 2 * radius))
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (32, 32, 40)
            cv2.circle(frame, (cx, cy), radius, (40, 180, 255), thickness=-1)
            cv2.putText(
                frame,
                f"{i}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate a synthetic demo clip.")
    parser.add_argument(
        "-o",
        "--output",
        default="samples/synthetic_30fps.mp4",
        help="Output path (default: samples/synthetic_30fps.mp4)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds")
    args = parser.parse_args()
    path = generate_synthetic_clip(
        args.output, fps=args.fps, duration_s=args.duration
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
