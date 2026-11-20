#!/usr/bin/env python3
"""Live metric depth + point cloud from a PS3 Eye, Depth Anything V2 on TensorRT.

    python live_depth.py --engine da2_metric_large.engine --calib calib.npz

Keys (in the image window): q / Esc quit, s save cloud (.ply) + depth (.npy),
c next colour map, i invert colour map.
"""
import argparse
import math
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Depth colour maps, in the order the "c" key cycles through them.
COLORMAPS = {
    name: getattr(cv2, "COLORMAP_" + name.upper())
    for name in ("inferno", "jet", "rainbow", "turbo", "viridis", "plasma", "magma", "hot",
                 "cool", "hsv", "bone", "ocean", "parula", "cividis", "twilight")
    if hasattr(cv2, "COLORMAP_" + name.upper())
}
COLORMAPS["gray"] = None  # plain greyscale


class Camera:
    """Grabs frames on a thread so the pipeline always gets the newest one."""

    def __init__(self, device, width, height, fps):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise SystemExit(f"Cannot open camera {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Camera opened but returned no frame")
        self.height, self.width = frame.shape[:2]
        self.frame, self.seq, self.running = frame, 1, True
        self.cond = threading.Condition()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.cond:
                self.frame, self.seq = frame, self.seq + 1
                self.cond.notify()

    def read(self, last_seq):
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq != last_seq, timeout=2.0):
                raise SystemExit("Camera stopped delivering frames")
            return self.frame, self.seq

    def close(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


class TRTDepth:
    """TensorRT engine with torch CUDA tensors as I/O buffers (no host copies)."""

    def __init__(self, engine_path):
        import tensorrt as trt

        torch.zeros(1, device="cuda")  # make sure torch owns a CUDA context first
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit("Failed to load engine (built with another TensorRT version/GPU?)")
        self.context = self.engine.create_execution_context()
        inp = out = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inp = name
            else:
                out = name
        self.input = torch.empty(tuple(self.engine.get_tensor_shape(inp)), dtype=torch.float32, device="cuda")
        self.output = torch.empty(tuple(self.engine.get_tensor_shape(out)), dtype=torch.float32, device="cuda")
        self.context.set_tensor_address(inp, self.input.data_ptr())
        self.context.set_tensor_address(out, self.output.data_ptr())
        self.height, self.width = self.input.shape[2:]

    def infer(self):
        # Same stream as torch, so this is ordered with the pre/post-processing ops.
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.output.view(self.height, self.width)


def load_intrinsics(args, cam_w, cam_h):
    """Returns K at camera resolution and optional undistortion maps."""
    if args.calib:
        data = np.load(args.calib)
        K, dist = data["K"].astype(np.float64), data["dist"].astype(np.float64)
        cw, ch = (int(v) for v in data["size"])
        if (cw, ch) != (cam_w, cam_h):
            K = K.copy()
            K[0] *= cam_w / cw
            K[1] *= cam_h / ch
        maps = cv2.initUndistortRectifyMap(K, dist, None, K, (cam_w, cam_h), cv2.CV_16SC2)
        return K, maps
    f = (cam_w / 2) / math.tan(math.radians(args.hfov) / 2)
    print(f"No --calib given: assuming {args.hfov:.0f} deg horizontal FOV (fx={f:.0f}px). Geometry will be approximate.")
    K = np.array([[f, 0, cam_w / 2], [0, f, cam_h / 2], [0, 0, 1]], dtype=np.float64)
    return K, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="da2_metric_large.engine")
    ap.add_argument("--device", default=0, type=lambda v: int(v) if v.isdigit() else v, help="camera index or /dev/videoN")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--calib", default=None, help="calib.npz from calibrate.py")
    ap.add_argument("--hfov", type=float, default=60.0, help="horizontal FOV in degrees, used only without --calib")
    ap.add_argument("--depth-scale", type=float, default=1.0, help="multiply depth to correct metric scale")
    ap.add_argument("--min-depth", type=float, default=0.15)
    ap.add_argument("--max-depth", type=float, default=8.0, help="metres; also the top of the colour scale")
    ap.add_argument("--edge-thresh", type=float, default=0.05,
                    help="drop points where depth jumps by more than this fraction between pixels (0 = keep all)")
    ap.add_argument("--colormap", choices=list(COLORMAPS), default="inferno", help="depth colour map (c cycles live)")
    ap.add_argument("--invert", action="store_true",
                    help="map far to the top of the colour map instead of near (i toggles live)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth point per axis")
    ap.add_argument("--point-size", type=float, default=2.0)
    ap.add_argument("--no-cloud", action="store_true", help="depth image only (no Open3D window)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available to PyTorch")
    dev = torch.device("cuda")

    model = TRTDepth(args.engine)
    mh, mw = model.height, model.width
    cam = Camera(args.device, args.width, args.height, args.fps)
    print(f"Camera {cam.width}x{cam.height}, model input {mw}x{mh}")

    K, undistort_maps = load_intrinsics(args, cam.width, cam.height)
    # Everything 3D runs at model resolution, so scale the intrinsics to it.
    fx, fy = K[0, 0] * mw / cam.width, K[1, 1] * mh / cam.height
    cx, cy = K[0, 2] * mw / cam.width, K[1, 2] * mh / cam.height
    v, u = torch.meshgrid(torch.arange(mh, device=dev), torch.arange(mw, device=dev), indexing="ij")
    # Y and Z flipped (OpenGL convention: +Y up, camera looks down -Z) so the cloud is upright in viewers.
    rays = torch.stack(((u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -torch.ones_like(u, dtype=torch.float32)), dim=-1)
    rays = rays.float()
    stride_mask = torch.zeros(mh, mw, dtype=torch.bool, device=dev)
    stride_mask[:: args.stride, :: args.stride] = True
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    vis = pcd = None
    if not args.no_cloud:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window("Point cloud", 960, 720)
        opt = vis.get_render_option()
        opt.point_size = args.point_size
        opt.background_color = np.array([0.05, 0.05, 0.07])
        pcd = o3d.geometry.PointCloud()
    cloud_added = False
    cmap_names = list(COLORMAPS)
    cmap_idx, invert = cmap_names.index(args.colormap), args.invert

    seq, fps, t_prev = 0, 0.0, time.perf_counter()
    try:
        while True:
            frame, seq = cam.read(seq)
            if undistort_maps is not None:
                frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)

            # Preprocess on GPU: BGR uint8 -> RGB float, resize, normalise.
            img = torch.from_numpy(frame).to(dev).permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255)
            rgb = F.interpolate(img, size=(mh, mw), mode="bilinear", align_corners=False)
            model.input.copy_((rgb - mean) / std)
            depth = model.infer() * args.depth_scale  # (mh, mw) metres

            valid = (depth > args.min_depth) & (depth < args.max_depth)
            if args.edge_thresh > 0:
                # Monocular depth smears across object boundaries; drop those "flying pixels".
                d = depth.view(1, 1, mh, mw)
                local_max = F.max_pool2d(d, 3, 1, 1)
                local_min = -F.max_pool2d(-d, 3, 1, 1)
                valid &= ((local_max - local_min) / d.clamp_min(1e-3)).view(mh, mw) < args.edge_thresh

            depth_u8 = (depth / args.max_depth).clamp(0, 1).mul(255).byte().cpu().numpy()

            points = colors = None
            if vis is not None:
                keep = valid & stride_mask
                points = (rays * depth.unsqueeze(-1))[keep].double().cpu().numpy()
                colors = rgb[0].permute(1, 2, 0)[keep].double().cpu().numpy()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                if not cloud_added and len(points):
                    vis.add_geometry(pcd)
                    ctr = vis.get_view_control()
                    ctr.set_front([0.0, 0.0, 1.0])
                    ctr.set_up([0.0, 1.0, 0.0])
                    ctr.set_zoom(0.5)
                    cloud_added = True
                elif cloud_added:
                    vis.update_geometry(pcd)
                if not vis.poll_events():
                    break
                vis.update_renderer()

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            # Default: near = top of the colour map (bright / red), far = bottom.
            scaled = depth_u8 if invert else 255 - depth_u8
            cmap = COLORMAPS[cmap_names[cmap_idx]]
            depth_vis = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR) if cmap is None else cv2.applyColorMap(scaled, cmap)
            depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            centre = float(depth[mh // 2, mw // 2])
            view = np.hstack((frame, depth_vis))
            cv2.drawMarker(view, (frame.shape[1] // 2, frame.shape[0] // 2), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
            cv2.putText(view, f"{fps:5.1f} FPS   centre {centre:.2f} m   {cmap_names[cmap_idx]}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("PS3 Eye | depth", view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(cmap_names)
            elif key == ord("i"):
                invert = not invert
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                np.save(f"depth_{stamp}.npy", depth.cpu().numpy())
                if points is not None and len(points):
                    o3d.io.write_point_cloud(f"cloud_{stamp}.ply", pcd)
                print(f"Saved depth_{stamp}.npy" + (f" and cloud_{stamp}.ply" if points is not None else ""))
    finally:
        cam.close()
        cv2.destroyAllWindows()
        if vis is not None:
            vis.destroy_window()


if __name__ == "__main__":
    main()
