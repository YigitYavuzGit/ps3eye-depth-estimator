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

In the image window: `c` cycles the depth colour map (jet, rainbow, turbo, ...), `i` inverts it,
`s` saves a `.ply` cloud and `.npy` depth, `q` quits. Pick the starting map with `--colormap jet`.
Run any script with `--help` for all options.

## Notes

- `large` gives the best quality; build `--encoder small` for the camera's full 60 FPS.
- Monocular metric depth is approximate. Correct a consistent offset with `--depth-scale`.
- If depth looks broken, rebuild the engine with `--fp32`.
- Engines are specific to the GPU and TensorRT version; rebuild after upgrading either.
