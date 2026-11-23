"""Maclss command-line interface."""

from __future__ import annotations

from pathlib import Path

import click

from maclss import __version__
from maclss.interpolators import get_interpolator
from maclss.pipeline import interpolate_video


@click.group()
@click.version_option(__version__, prog_name="maclss")
def main() -> None:
    """Maclss — frame interpolation for video (and later CrossOver capture)."""


@main.command("interpolate")
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output video path (.mp4 / .mov / .avi).",
)
@click.option(
    "--factor",
    default=2,
    show_default=True,
    type=click.IntRange(min=2),
    help="FPS multiply factor. 2 inserts one frame between each pair (~30→60).",
)
@click.option(
    "--backend",
    default="optical-flow",
    show_default=True,
    type=click.Choice(["optical-flow", "rife"], case_sensitive=False),
    help="Interpolator backend. rife is a stub until model weights are bundled.",
)
def interpolate_cmd(
    input_path: Path,
    output_path: Path,
    factor: int,
    backend: str,
) -> None:
    """Interpolate INPUT to roughly FACTOR× FPS and write OUTPUT."""
    try:
        interpolator = get_interpolator(backend)
    except NotImplementedError as exc:
        raise click.ClickException(str(exc)) from exc

    def _progress(done: int, total: int) -> None:
        # Simple carriage-return progress; quiet enough for smoke tests.
        pct = (100 * done) // total if total else 0
        click.echo(f"\rFrames: {done}/{total} ({pct}%)", nl=False)

    click.echo(
        f"Interpolating {input_path} → {output_path} "
        f"(factor={factor}, backend={backend})"
    )
    stats = interpolate_video(
        input_path=input_path,
        output_path=output_path,
        interpolator=interpolator,
        factor=factor,
        progress=_progress,
    )
    click.echo()  # finish progress line
    click.echo(
        f"Done. {stats['frames_in']} → {stats['frames_out']} frames, "
        f"{stats['input_fps']:.2f} → {stats['output_fps']:.2f} FPS "
        f"(backend={stats['backend']})"
    )


# --- Extension notes (not wired) -------------------------------------------
# Live CrossOver / ScreenCaptureKit capture belongs in a later slice.
# Keep this CLI focused on offline video so quality can be judged first.
# Suggested future commands (do not implement here):
#   maclss capture --window "CrossOver" --preview
#   maclss capture --window "CrossOver" -o out.mp4


if __name__ == "__main__":
    main()
