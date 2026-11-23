# Maclss

Frame interpolation for offline video testing (and later CrossOver / game capture on macOS).

**v1 deliverable:** take a ~30 FPS video, insert one interpolated frame between each consecutive pair, and write a playable ~60 FPS result.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.10+ and a working OpenCV build (pulled in via `opencv-python-headless`).

## Usage

```bash
# Interpolate any mp4/mov (or avi) to ~2× FPS
maclss interpolate input.mp4 -o output_60fps.mp4

# Explicit factor / backend
maclss interpolate input.mp4 -o out.mp4 --factor 2 --backend optical-flow
```

### Demo / smoke test without your own footage

```bash
# 1. Generate a short synthetic ~30 FPS clip
python -m maclss.demo -o samples/synthetic_30fps.mp4

# 2. Interpolate it
maclss interpolate samples/synthetic_30fps.mp4 -o samples/synthetic_60fps.mp4

# 3. Or run the automated smoke tests
pytest -q
```

The smoke tests assert that frame count becomes `2N - 1` and output FPS is ~2× input FPS.

## Interpolator

| Backend | Status | Notes |
|---------|--------|-------|
| `optical-flow` (default) | **Working** | OpenCV Farneback dense flow + bidirectional warp/blend. Real motion-compensated interpolation (not frame duplication). |
| `rife` | Stub | Extension point for a RIFE-style neural model. Optional extra: `pip install 'maclss[rife]'`. Implement loading in `maclss/interpolators/rife.py`. |

Backends share the `FrameInterpolator` protocol in `maclss/interpolators/base.py`, so a better model can plug in without changing the video pipeline.

## Project layout

```
maclss/
  cli.py                 # maclss interpolate …
  pipeline.py            # read → interpolate → write
  demo.py                # synthetic clip generator
  interpolators/
    base.py              # protocol
    optical_flow.py      # default backend
    rife.py              # neural stub / future hook
```

## Out of scope (v1)

- CrossOver / ScreenCaptureKit live capture (next slice)
- Upscaling
- Multi-game profiles

Suggested future CLI surface (not implemented): `maclss capture --window "CrossOver" --preview`.

## License

MIT
