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
  and a point cloud preview. In the point cloud view, drag to orbit,
  right-drag (or middle-drag, Shift+drag, arrow keys) to pan, and scroll to zoom.
- **Colour map**: click a swatch (inferno, jet, rainbow, turbo, viridis, ... and grey).
- **Range slider**: drag the near and far handles to set the depth range. Colours span it, and
  everything outside is greyed out in the depth view and removed from the point cloud.
  Starting values: `--min-depth`, `--max-depth`; slider length: `--range-limit` (10 m).
- **Invert**, **Auto range** (stretch colours to what is actually inside the range),
  **Cloud colour: depth**, **Reset view**, **Save** (`.ply` cloud, `.npy` depth, `.png` view).

Hover over a view to read the distance under the cursor in the status bar. The window can be
resized, maximised or made fullscreen (**Fullscreen** button, `f`, or `--fullscreen`); views scale to fit.

The **Layout** button (`l`, `--layout`) switches between `grid` (even grid), `stack` (views on top of
each other) and `focus` (one big view using the full height, the rest in a column on the right).
In focus, click a small view to make it the big one; double-clicking any view enlarges it.

Keys: `1`-`6` toggle views, `c` next colour map, `i` invert, `a` auto range, `r` reset cloud view,
`+`/`-` zoom, `[` `]` far limit, `;` `'` near limit, `l` layout, `f` fullscreen, `s` save, `q` quit. Starting state: `--views camera,depth,cloud --colormap jet`.

The confidence view is an estimate, not a model output: Depth Anything V2 does not predict
confidence, so it is derived from depth discontinuities and frame-to-frame stability.
`--min-conf 0.5` drops low-confidence points from the cloud. `--open3d` (needs `pip install open3d`) additionally opens a
full Open3D point cloud window.
Run any script with `--help` for all options.

## macOS (Apple silicon) and machines without TensorRT

The viewer also has a PyTorch backend that needs no engine build. It is picked automatically when
there is no NVIDIA GPU or no engine file, and uses the Apple GPU (MPS) on a Mac.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # TensorRT and ONNX are skipped on macOS
python live_depth.py --encoder small   # or base / large; add --half to try half precision
```

- macOS has no PS3 Eye driver, so use the built-in camera or a USB webcam (`--device 0`, `1`, ...).
  Allow camera access for your terminal when macOS asks. Widescreen cameras are centre-cropped
  to 4:3.
- Calibrate that camera with `calibrate.py` (it applies the same crop); a calibration from the
  PS3 Eye does not carry over.
- The FoundationPose bridge (`--pose`) needs an NVIDIA machine for the server side.
- Force a backend with `--backend torch` or `--backend tensorrt`.

## Tuning GPU load, speed and accuracy

By default the viewer processes every camera frame (up to 60 per second), which keeps the GPU busy.

| Knob | How | Effect |
|---|---|---|
| Frame-rate cap | **Max FPS** button, `t`, or `--max-fps 15` | GPU load drops roughly in proportion; no quality change per frame |
| Model size | `build_engine.py --encoder small` / `base` / `large` | Smaller is much lighter and uses less GPU memory, with softer detail and less stable scale |
| Input resolution | `build_engine.py --height 294 --width 392` (multiples of 14, keep 3:4) | Fewer pixels to process; coarser depth map and point cloud |
| Camera rate | `--fps 30` | Fewer frames arrive in the first place |
| Views | turn off Point cloud / Normals / Confidence | Skips their rendering work |
| Cloud density | `--stride 2` | A quarter of the points |

Engines for different settings can live side by side: add `--out name.engine` when building and
pass that file to `live_depth.py --engine`.

## 6D object pose with FoundationPose (optional)

The viewer can feed a local [FoundationPose](https://github.com/NVlabs/FoundationPose) install and
show the tracked object pose live. FoundationPose keeps its own environment; the two processes
talk over a local socket.

**1. In the FoundationPose environment**, start the bridge server (it only needs FoundationPose's
own dependencies):

```bash
# model-based: you have a mesh of the object, in metres (--mesh-scale 0.001 for millimetres)
python /path/to/ps3eye-depth-estimator/fp_server.py --fp-dir /path/to/FoundationPose --mesh object.obj

# model-free: a folder of reference views in FoundationPose's ref-view format
python /path/to/ps3eye-depth-estimator/fp_server.py --fp-dir /path/to/FoundationPose --ref-views /path/to/ob_0000001
```

**2. In this project's environment**, start the viewer with `--pose`:

```bash
python live_depth.py --engine da2_metric_large.engine --calib calib.npz --pose
```

Click **Select object** (`p`), then click the object in the camera view. Segment Anything turns
the click into a mask (shown in green), FoundationPose estimates the pose from it and then tracks
frame to frame. The oriented box and x/y/z axes (red/green/blue) are drawn on the camera and
depth views; the status bar shows the position in metres and FoundationPose's time per frame.
**Stop pose** (`o`) stops tracking; selecting again re-initialises it.

Things to know:

- FoundationPose is designed for depth-sensor depth. Here it gets estimated depth, whose absolute
  scale can be several percent off and which flickers slightly. Because the mesh has a true size,
  a scale error shows up as a worse fit and a wrong distance. Calibrate the camera and set
  `--depth-scale` against a tape measure before judging the results.
- Model-free mode trains FoundationPose's neural object field on the reference views the first
  time (slow, needs its BundleSDF/Kaolin parts) and caches `model/model.obj` in that folder. The
  reference views need a known camera pose per view (`cam_in_ob/`), which this viewer cannot
  produce, so they have to come from elsewhere.
- The first click downloads the Segment Anything checkpoint (`--sam-model`, default
  `facebook/sam-vit-base`).
- Depth, Segment Anything and FoundationPose all share one GPU, so expect a lower frame rate
  than depth alone. `build_engine.py --encoder small` frees some of it.
- `--pose-host` / `--pose-port` point the viewer at a server elsewhere; `fp_server.py --host`
  controls which interface it listens on (this machine only by default).

## Trying the data in Delta DIAVision

**Save** (`s`) also writes a `diavision_<time>/` folder in the formats DIAVision's VirtualCamera
loads (Manage > Platform Settings > VirtualCamera): `image.png` (8-bit), `depth_mm.png` (16-bit
depth map, millimetres, 0 = no data) and `cloud_mm.pcd` (points in millimetres, x right, y down,
z forward from the lens). This is a file hand-off for testing, not a live connection.

## Live stream for other programs

`python live_depth.py --serve` streams each frame on a local TCP port (5600): grey image, colour
image, 16-bit depth in millimetres and a per-pixel XYZ point cloud in millimetres, all on the
camera's 640x480 grid. The format is documented at the top of `frame_server.py`.
`python stream_client.py` is a reference client that prints what arrives. The stream is meant
for the DIAVision plug-in in `diavision_plugin/` (see its README), which is still in progress.

The viewer also opens the camera through DirectShow on Windows, so it can run next to DIAVision;
the PS3 Eye needs a DirectShow driver there, and TensorRT engines have to be rebuilt on Windows.

## Notes

- The viewer window uses pygame-ce (SDL; imported as `pygame`), not OpenCV's own window, whose Qt backend zooms the whole
  image on mouse wheel. `calibrate.py` still uses an OpenCV window.

- `large` gives the best quality; build `--encoder small` for the camera's full 60 FPS.
- Monocular metric depth is approximate. Correct a consistent offset with `--depth-scale`.
- If depth looks broken, rebuild the engine with `--fp32`.
- Engines are specific to the GPU and TensorRT version; rebuild after upgrading either.
