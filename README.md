# ps3eye-depth-estimator

Live metric depth and coloured point cloud from a single PS3 Eye camera on Ubuntu,
using Depth Anything V2 (metric, indoor) accelerated with TensorRT on an NVIDIA GPU.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. Export ONNX and build the TensorRT engine (once; --encoder small|base|large)
python build_engine.py

# 2. Calibrate the camera with a checkerboard (once; writes calib.npz)
python calibrate.py --cols 9 --rows 6 --square 0.025

# 3. Run live
python live_depth.py --engine da2_metric_large.engine --calib calib.npz
```

Everything is selectable with the mouse in the window's toolbar:

- **Views** (any combination, tiled): camera, depth, intensity, confidence, surface normals,
  and a point cloud preview. In the point cloud view, drag to orbit and scroll to zoom.
- **Colour map**: click a swatch (inferno, jet, rainbow, turbo, viridis, ... and grey).
- **Range slider**: drag the near and far handles to set the depth range. Colours span it, and
  everything outside is greyed out in the depth view and removed from the point cloud.
  Starting values: `--min-depth`, `--max-depth`; slider length: `--range-limit` (10 m).
- **Invert**, **Auto range** (stretch colours to what is actually inside the range),
  **Cloud colour: depth**, **Reset view**, **Save** (`.ply` cloud, `.npy` depth, `.png` view).

Hover over a view to read the distance under the cursor in the status bar. The window can be
resized, maximised or made fullscreen (**Fullscreen** button, `f`, or `--fullscreen`); views scale to fit.

Keys: `1`-`6` toggle views, `c` next colour map, `i` invert, `a` auto range, `r` reset cloud view,
`+`/`-` zoom, `[` `]` far limit, `;` `'` near limit, `f` fullscreen, `s` save, `q` quit. Starting state: `--views camera,depth,cloud --colormap jet`.

The confidence view is an estimate, not a model output: Depth Anything V2 does not predict
confidence, so it is derived from depth discontinuities and frame-to-frame stability.
`--min-conf 0.5` drops low-confidence points from the cloud. `--open3d` additionally opens a
full Open3D point cloud window.
Run any script with `--help` for all options.

## Notes

- `large` gives the best quality; build `--encoder small` for the camera's full 60 FPS.
- Monocular metric depth is approximate. Correct a consistent offset with `--depth-scale`.
- If depth looks broken, rebuild the engine with `--fp32`.
- Engines are specific to the GPU and TensorRT version; rebuild after upgrading either.
