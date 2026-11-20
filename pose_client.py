"""Viewer side of the FoundationPose bridge: server connection, click-to-segment, pose overlay."""
import threading
import time
from multiprocessing.connection import Client

import cv2
import numpy as np
import torch

import pose_protocol as proto


class PoseTracker:
    """Talks to fp_server.py on a background thread so the viewer never waits for a pose.
    Only the newest frame is kept; frames that arrive while a request is in flight are dropped."""

    def __init__(self, host, port):
        self.address = (host, port)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.job = None            # (generation, command, arrays)
        self.generation = 0        # bumped by register/stop so stale replies are ignored
        self.registered = False
        self.pose = None           # 4x4 object-in-camera, or None
        self.ms = 0.0
        self.extents = self.to_origin = None
        self.name = ""
        self.connected = False
        self.status = "connecting to fp_server.py"
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def register(self, rgb, depth, K, mask):
        with self.lock:
            self.generation += 1
            self.registered, self.pose = False, None
            self.job = (self.generation, "register", (rgb, depth, K, mask.astype(np.uint8)))
            if self.connected:
                self.status = "estimating first pose..."
        self.wake.set()

    def track(self, rgb, depth, K):
        with self.lock:
            if not self.registered or (self.job and self.job[1] == "register"):
                return
            self.job = (self.generation, "track", (rgb, depth, K))
        self.wake.set()

    def stop(self):
        with self.lock:
            self.generation += 1
            self.job, self.registered, self.pose = None, False, None
            if self.connected:
                self.status = "idle - Select object, then click it"

    def close(self):
        self.running = False
        self.wake.set()

    def _loop(self):
        while self.running:
            try:
                conn = Client(self.address)
            except OSError:
                self.status = "waiting for fp_server.py"
                time.sleep(1.0)
                continue
            try:
                hello, (self.extents, self.to_origin) = proto.recv(conn)
                self.name = hello.get("name", "")
                with self.lock:
                    self.connected = True
                    if self.job is None:
                        self.status = "idle - Select object, then click it"
                while self.running:
                    self.wake.wait(0.2)
                    with self.lock:
                        job, self.job = self.job, None
                        self.wake.clear()
                    if job is None:
                        continue
                    generation, cmd, arrays = job
                    proto.send(conn, {"cmd": cmd}, *arrays)
                    reply, out = proto.recv(conn)
                    with self.lock:
                        if generation != self.generation:
                            continue  # stopped or re-registered meanwhile
                        if reply.get("ok"):
                            self.pose, self.ms, self.registered = out[0], reply.get("ms", 0.0), True
                            self.status = "tracking"
                        else:
                            self.pose, self.registered = None, False
                            self.status = "error: " + str(reply.get("error", "?"))[:70]
            except (EOFError, ConnectionError, OSError, ValueError):
                pass
            finally:
                conn.close()
                with self.lock:
                    self.connected, self.registered, self.pose = False, False, None
                    self.status = "fp_server.py disconnected"


class Segmenter:
    """Click-to-segment with Segment Anything (via transformers), loaded on first use."""

    def __init__(self, model_name, device):
        self.model_name, self.device = model_name, device
        self.model = self.processor = None

    def __call__(self, rgb, x, y):
        """rgb: (H,W,3) uint8, click at pixel (x, y). Returns a bool mask (H,W)."""
        if self.model is None:
            from transformers import SamModel, SamProcessor

            print(f"Loading {self.model_name} (downloaded on first use)...")
            self.processor = SamProcessor.from_pretrained(self.model_name)
            self.model = SamModel.from_pretrained(self.model_name).to(self.device).eval()
        inputs = self.processor(images=rgb, input_points=[[[float(x), float(y)]]], return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        post = self.processor.image_processor.post_process_masks
        try:
            masks = post(out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        except (TypeError, KeyError):  # newer processors take only the original sizes
            masks = post(out.pred_masks.cpu(), inputs["original_sizes"].cpu())
        # SAM proposes three masks (part / object / whole); keep the one it scores highest.
        best = int(out.iou_scores[0, 0].argmax())
        return masks[0][0, best].numpy().astype(bool)


# Corners indexed by bits (x, y, z); edges join corners that differ in one bit.
_BOX_EDGES = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def draw_pose(img, pose, K, extents, to_origin):
    """Draws the object's oriented bounding box and its x/y/z axes (red/green/blue) on a BGR image."""
    centre = pose @ np.linalg.inv(to_origin)  # box-centred object frame in camera coordinates
    half = np.asarray(extents, np.float64) / 2

    def project(points):
        cam = points @ centre[:3, :3].T + centre[:3, 3]
        if (cam[:, 2] <= 1e-3).any():
            return None
        uv = cam @ np.asarray(K, np.float64).T
        return np.clip(uv[:, :2] / uv[:, 2:], -1e5, 1e5).round().astype(int)

    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64) * half
    uv = project(corners)
    if uv is None:
        return
    for a, b in _BOX_EDGES:
        cv2.line(img, tuple(map(int, uv[a])), tuple(map(int, uv[b])), (0, 255, 0), 2, cv2.LINE_AA)
    length = float(max(half))
    axes = project(np.vstack((np.zeros(3), np.eye(3) * length)))
    if axes is not None:
        for i, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(img, tuple(map(int, axes[0])), tuple(map(int, axes[i + 1])), colour, 3, cv2.LINE_AA)
