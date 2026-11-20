#!/usr/bin/env python3
"""Live metric depth + point cloud from a PS3 Eye, Depth Anything V2 on TensorRT.

    python live_depth.py --engine da2_metric_large.engine --calib calib.npz

Everything is selectable with the mouse in the window's toolbar: which views are
shown (camera, depth, intensity, confidence, normals, point cloud), the colour map,
invert, auto range, the point cloud colouring, and the depth range (drag the two
handles of the range slider). In the point cloud view, drag to orbit, right-drag
(or middle-drag, Shift+drag, arrow keys) to pan, and scroll to zoom.

Keys: 1-6 toggle views, c next colour map, i invert, a auto range, r reset cloud
view, +/- zoom cloud, [ ] move the far limit, ; ' move the near limit,
l next layout, f fullscreen, s save (.ply, .npy, .png), q / Esc quit (Esc leaves fullscreen first).

The window can be resized, maximised or made fullscreen; the views scale to fit.
Layouts: grid, stack (vertical), focus (one big view; click a small view or
double-click any view to enlarge it).
"""
import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Every colour map OpenCV ships, common ones first. "gray" is plain greyscale.
_CMAP_NAMES = ("inferno", "jet", "rainbow", "turbo", "viridis", "plasma", "magma", "hot", "cool", "hsv",
               "bone", "ocean", "parula", "cividis", "twilight", "twilight_shifted", "autumn", "winter",
               "spring", "summer", "pink", "deepgreen")
COLORMAPS = {n: getattr(cv2, "COLORMAP_" + n.upper()) for n in _CMAP_NAMES if hasattr(cv2, "COLORMAP_" + n.upper())}
COLORMAPS["gray"] = None

LAYOUTS = ("grid", "stack", "focus")
PANES = ("camera", "depth", "intensity", "confidence", "normals", "cloud")
PANE_LABELS = {"camera": "Camera", "depth": "Depth", "intensity": "Intensity",
               "confidence": "Confidence", "normals": "Normals", "cloud": "Point cloud"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
ACCENT = (200, 140, 40)


def colormap_lut(name):
    """(256, 3) BGR lookup table for a colour map."""
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    cmap = COLORMAPS[name]
    lut = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR) if cmap is None else cv2.applyColorMap(ramp, cmap)
    return lut.reshape(256, 3)


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
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            raise SystemExit(f"{engine_path} not found - run build_engine.py first")
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


class Gui:
    """Toolbar, view layout and mouse state for the single OpenCV window."""

    MIN_CONTENT = 240
    ROW = 30
    STATUS = 24

    def __init__(self, args):
        self.width = 1280
        self.fullscreen = args.fullscreen
        self.layout_mode = args.layout
        self.main = None  # the big view in the focus layout
        self.promoted = None
        self.panes = {p: p in args.views for p in PANES}
        self.cmap = args.colormap
        self.invert = args.invert
        self.auto_range = args.auto_range
        self.cloud_depth_colour = False
        self.yaw = self.pitch = 0.0
        self.zoom = 1.0
        self.mouse = (-1, -1)
        self.drag = None
        self.pan = [0.0, 0.0]        # cloud view shift in metres (right, down)
        self.pan_drag = None
        self.metres_per_px = 0.005   # at the orbit centre; updated every frame by the renderer
        # Depth range in metres: colours span it and everything outside is clipped.
        self.limit = max(args.range_limit, args.max_depth)
        self.near, self.far = args.min_depth, args.max_depth
        self.slider_drag = None
        self.pending = []  # one-shot actions for the main loop: "save", "quit"
        self.cells = {}
        self.luts = {name: colormap_lut(name) for name in COLORMAPS}
        self._build_buttons()
        self.min_width = self.row1_end + 6  # the top button row needs this much

    # ---- toolbar -------------------------------------------------------------
    def _build_buttons(self):
        self.buttons = []  # (x0, y0, x1, y1, kind, value, label)
        x, pad = 6, 9

        def add(kind, value, label, gap=4):
            nonlocal x
            w = cv2.getTextSize(label, FONT, 0.45, 1)[0][0] + 2 * pad
            self.buttons.append((x, 3, x + w, self.ROW - 3, kind, value, label))
            x += w + gap

        for i, p in enumerate(PANES):
            add("pane", p, PANE_LABELS[p], gap=18 if i == len(PANES) - 1 else 4)
        add("toggle", "invert", "Invert")
        add("toggle", "auto_range", "Auto range")
        add("toggle", "cloud_depth_colour", "Cloud colour: depth", gap=18)
        add("action", "reset", "Reset view")
        add("action", "save", "Save", gap=18)
        add("cycle", "layout", "Layout: focus")
        add("toggle", "fullscreen", "Fullscreen")
        add("action", "quit", "Quit")
        self.row1_end = x

        names = list(COLORMAPS)
        per_row = math.ceil(len(names) / 2)
        bw = (self.width - 12) // per_row
        for i, name in enumerate(names):
            r, c = divmod(i, per_row)
            x0, y0 = 6 + c * bw, self.ROW * (1 + r) + 2
            self.buttons.append((x0, y0, x0 + bw - 3, y0 + self.ROW - 4, "cmap", name, name))
        self.slider_y0 = self.ROW * 3 + 4
        self.slider_x0, self.slider_x1 = 70, self.width - 200
        self.toolbar_h = self.ROW * 4 + 6

    # ---- range slider --------------------------------------------------------
    def _slider_x(self, metres):
        return int(round(self.slider_x0 + (self.slider_x1 - self.slider_x0) * metres / self.limit))

    def set_range(self, near=None, far=None):
        gap = 0.05
        if near is not None:
            self.near = round(min(max(near, 0.0), self.far - gap), 2)
        if far is not None:
            self.far = round(min(max(far, self.near + gap), self.limit), 2)

    def _slider_move(self, x):
        metres = (x - self.slider_x0) / (self.slider_x1 - self.slider_x0) * self.limit
        self.set_range(**{self.slider_drag: metres})

    def _draw_slider(self, bar):
        y0, y1 = self.slider_y0 + 7, self.slider_y0 + self.ROW - 7
        xn, xf = self._slider_x(self.near), self._slider_x(self.far)
        cv2.putText(bar, "Range", (10, y1 - 3), FONT, 0.45, (205, 205, 205), 1, cv2.LINE_AA)
        cv2.rectangle(bar, (self.slider_x0, y0), (self.slider_x1, y1), (62, 62, 62), -1)
        if xf > xn:  # the selected span, painted with the colours it maps to
            ramp = np.linspace(0, 255, xf - xn).astype(np.int64)
            bar[y0:y1 + 1, xn:xf] = self.luts[self.cmap][ramp if self.invert else 255 - ramp][None]
        for metre in range(int(self.limit) + 1):
            x = self._slider_x(metre)
            cv2.line(bar, (x, y1 + 1), (x, y1 + 4), (150, 150, 150), 1)
        for x in (xn, xf):
            cv2.rectangle(bar, (x - 4, y0 - 4), (x + 4, y1 + 4), (255, 255, 255), -1)
            cv2.rectangle(bar, (x - 4, y0 - 4), (x + 4, y1 + 4), (40, 40, 40), 1)
        cv2.putText(bar, f"{self.near:.2f} - {self.far:.2f} m", (self.slider_x1 + 18, y1 - 3),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def resize(self, width):
        if width != self.width:
            self.width = width
            self._build_buttons()

    def _active(self, kind, value):
        if kind == "pane":
            return self.panes[value]
        if kind == "toggle":
            return getattr(self, value)
        return False

    def draw_toolbar(self):
        bar = np.full((self.toolbar_h, self.width, 3), 30, np.uint8)
        for x0, y0, x1, y1, kind, value, text in self.buttons:
            if kind == "cmap":
                swatch = self.luts[value].reshape(1, 256, 3)
                bar[y0:y1, x0:x1] = cv2.resize(swatch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                short = text.replace("twilight_shifted", "twilight_s")
                label(bar, short, (x0 + 5, y1 - 8), 0.42)
                if value == self.cmap:
                    cv2.rectangle(bar, (x0 - 1, y0 - 1), (x1, y1), (255, 255, 255), 2)
                continue
            if kind == "cycle":
                text = f"Layout: {self.layout_mode}"
            on = self._active(kind, value)
            cv2.rectangle(bar, (x0, y0), (x1, y1), ACCENT if on else (62, 62, 62), -1)
            cv2.rectangle(bar, (x0, y0), (x1, y1), (110, 110, 110), 1)
            cv2.putText(bar, text, (x0 + 9, y1 - 8), FONT, 0.45, (255, 255, 255) if on else (205, 205, 205), 1, cv2.LINE_AA)
        self._draw_slider(bar)
        return bar

    def _click(self, kind, value):
        if kind == "pane":
            self.toggle_pane(value)
        elif kind == "toggle":
            setattr(self, value, not getattr(self, value))
        elif kind == "cmap":
            self.cmap = value
        elif kind == "cycle":
            self.next_layout()
        elif value == "reset":
            self.reset_view()
        else:
            self.pending.append(value)

    def toggle_pane(self, pane):
        self.panes[pane] = not self.panes[pane]
        if not any(self.panes.values()):
            self.panes[pane] = True  # always keep one view

    def next_cmap(self):
        names = list(COLORMAPS)
        self.cmap = names[(names.index(self.cmap) + 1) % len(names)]

    def pan_by(self, dx_px, dy_px):
        self.pan[0] += dx_px * self.metres_per_px
        self.pan[1] += dy_px * self.metres_per_px

    def reset_view(self):
        self.yaw = self.pitch = 0.0
        self.pan = [0.0, 0.0]
        self.zoom = 1.0

    def zoom_by(self, factor):
        self.zoom = min(max(self.zoom * factor, 0.15), 4.0)

    # ---- layout and mouse ----------------------------------------------------
    def layout(self, content_h, aspect):
        """Places the enabled views below the toolbar, each as large as fits at the camera's
        aspect ratio. Returns {pane: (x, y, w, h)} in canvas pixels.

        grid:  even grid (1, 2, 2x2, 3x2), starting a new row rather than squeezing a row
        stack: one column, views on top of each other
        focus: one big view using the full height, the others as a column on the right
        """
        active = [p for p in PANES if self.panes[p]]
        n = len(active)
        width, top = self.width, self.toolbar_h

        def fit(x, y, w, h):  # largest aspect-correct rectangle centred in a slot
            cw = min(w, h * aspect)
            ch = cw / aspect
            return int(x + (w - cw) / 2), int(y + (h - ch) / 2), int(cw), int(ch)

        cells = {}
        if self.layout_mode == "focus" and n > 1:
            if self.main not in active:
                self.main = active[0]
            others = [p for p in active if p != self.main]
            slot_h = content_h / len(others)
            side_w = min(slot_h * aspect, width * 0.3)
            cells[self.main] = fit(0, top, width - side_w, content_h)
            for i, p in enumerate(others):
                cells[p] = fit(width - side_w, top + i * slot_h, side_w, slot_h)
        else:
            cols = 1 if self.layout_mode == "stack" else math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            slot_w, slot_h = width / cols, content_h / rows
            for i, p in enumerate(active):
                r, c = divmod(i, cols)
                in_row = min(cols, n - r * cols)          # centre a short last row
                x_off = (width - in_row * slot_w) / 2
                cells[p] = fit(x_off + c * slot_w, top + r * slot_h, slot_w, slot_h)
        self.cells = cells
        return cells

    def next_layout(self):
        self.layout_mode = LAYOUTS[(LAYOUTS.index(self.layout_mode) + 1) % len(LAYOUTS)]

    def pane_at(self, x, y):
        for p, (cx, cy, cw, ch) in self.cells.items():
            if cx <= x < cx + cw and cy <= y < cy + ch:
                return p, (x - cx) / cw, (y - cy) / ch
        return None, 0.0, 0.0

    def on_mouse(self, event, x, y, flags, _param=None):
        self.mouse = (x, y)
        pane = self.pane_at(x, y)[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.promoted = None
            if self.slider_y0 <= y < self.toolbar_h:
                # Grab whichever handle is closer to the click.
                dn, df = abs(x - self._slider_x(self.near)), abs(x - self._slider_x(self.far))
                self.slider_drag = "near" if dn < df or (dn == df and x < self._slider_x(self.near)) else "far"
                self._slider_move(x)
            elif y < self.toolbar_h:
                for x0, y0, x1, y1, kind, value, _ in self.buttons:
                    if x0 <= x <= x1 and y0 <= y <= y1:
                        self._click(kind, value)
                        break
            elif self.layout_mode == "focus" and pane and pane != self.main:
                self.main = self.promoted = pane  # clicking a small view makes it the big one
            elif pane == "cloud":
                # Shift + left drag pans, plain left drag orbits.
                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    self.pan_drag = (x, y)
                else:
                    self.drag = (x, y)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN) and pane == "cloud":
            self.pan_drag = (x, y)  # right or middle drag pans
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            self.pan_drag = None
        elif event == cv2.EVENT_MOUSEMOVE and self.pan_drag and flags & (
                cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON | cv2.EVENT_FLAG_MBUTTON):
            self.pan_by(x - self.pan_drag[0], y - self.pan_drag[1])
            self.pan_drag = (x, y)
        elif event == cv2.EVENT_LBUTTONDBLCLK and pane:
            # Double-click a view to enlarge it; double-click the big view to go back to the grid.
            if pane == self.promoted:
                pass  # second half of the click that just enlarged this view
            elif self.layout_mode == "focus" and pane == self.main:
                self.layout_mode = "grid"
            else:
                self.layout_mode, self.main = "focus", pane
        elif event == cv2.EVENT_MOUSEMOVE and self.slider_drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self._slider_move(x)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag and flags & cv2.EVENT_FLAG_LBUTTON:
            self.yaw -= (x - self.drag[0]) * 0.008
            self.pitch = min(max(self.pitch + (y - self.drag[1]) * 0.008, -1.45), 1.45)
            self.drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = self.slider_drag = self.pan_drag = None
        elif event == cv2.EVENT_MOUSEWHEEL and pane == "cloud":
            # The wheel delta is carried in the sign of flags (positive = scroll up / away).
            self.zoom_by(0.88 if flags > 0 else 1 / 0.88)


class Window:
    """The application window, on SDL through pygame. OpenCV's own window is not used: its Qt
    backend zooms the whole image on mouse wheel and offers no way to turn that off, and it
    does not report the window size reliably."""

    DOUBLE_CLICK = 0.35  # seconds

    def __init__(self, title, gui):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg, self.gui = pygame, gui
        pygame.display.init()
        pygame.display.set_caption(title)
        pygame.key.set_repeat(300, 40)  # held arrow keys keep panning
        self.windowed_size = (1280, 900)
        self.fullscreen = None
        self.scale = (1.0, 1.0)
        self.last_click = (0.0, 0, 0)
        self.set_fullscreen(gui.fullscreen)

    def set_fullscreen(self, on):
        if on == self.fullscreen:
            return
        pg = self.pg
        if on:
            if self.fullscreen is not None:
                self.windowed_size = self.screen.get_size()
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)  # desktop resolution
        else:
            self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
            if self.screen.get_size() != self.windowed_size:
                # Straight out of fullscreen SDL can keep the fullscreen size; asking again restores it.
                self.screen = pg.display.set_mode(self.windowed_size, pg.RESIZABLE)
        self.fullscreen = on

    def size(self):
        self.screen = self.pg.display.get_surface()
        return self.screen.get_size()

    def _canvas_xy(self, pos):
        return int(pos[0] * self.scale[0]), int(pos[1] * self.scale[1])

    def poll(self):
        """Forwards mouse events to the GUI. Returns (typed characters, window closed)."""
        pg, gui = self.pg, self.gui
        keys, closed = [], False
        for e in pg.event.get():
            if e.type == pg.QUIT:
                closed = True
            elif e.type == pg.KEYDOWN:
                arrows = {pg.K_LEFT: "left", pg.K_RIGHT: "right", pg.K_UP: "up", pg.K_DOWN: "down"}
                keys.append("\x1b" if e.key == pg.K_ESCAPE else arrows.get(e.key, e.unicode))
            elif e.type == pg.MOUSEBUTTONDOWN and e.button in (2, 3):
                event = cv2.EVENT_MBUTTONDOWN if e.button == 2 else cv2.EVENT_RBUTTONDOWN
                gui.on_mouse(event, *self._canvas_xy(e.pos), 0)
            elif e.type == pg.MOUSEBUTTONUP and e.button in (2, 3):
                event = cv2.EVENT_MBUTTONUP if e.button == 2 else cv2.EVENT_RBUTTONUP
                gui.on_mouse(event, *self._canvas_xy(e.pos), 0)
            elif e.type == pg.MOUSEBUTTONDOWN and e.button == 1:
                x, y = self._canvas_xy(e.pos)
                now = time.perf_counter()
                t, px, py = self.last_click
                double = now - t < self.DOUBLE_CLICK and abs(x - px) < 6 and abs(y - py) < 6
                self.last_click = (0.0 if double else now, x, y)
                shift = cv2.EVENT_FLAG_SHIFTKEY if pg.key.get_mods() & pg.KMOD_SHIFT else 0
                gui.on_mouse(cv2.EVENT_LBUTTONDBLCLK if double else cv2.EVENT_LBUTTONDOWN, x, y,
                             cv2.EVENT_FLAG_LBUTTON | shift)
            elif e.type == pg.MOUSEBUTTONUP and e.button == 1:
                gui.on_mouse(cv2.EVENT_LBUTTONUP, *self._canvas_xy(e.pos), 0)
            elif e.type == pg.MOUSEMOTION:
                held = ((cv2.EVENT_FLAG_LBUTTON if e.buttons[0] else 0) | (cv2.EVENT_FLAG_MBUTTON if e.buttons[1] else 0)
                        | (cv2.EVENT_FLAG_RBUTTON if e.buttons[2] else 0))
                gui.on_mouse(cv2.EVENT_MOUSEMOVE, *self._canvas_xy(e.pos), held)
            elif e.type == pg.MOUSEWHEEL and e.y:
                # Applies to the view under the pointer (last position the GUI saw).
                gui.on_mouse(cv2.EVENT_MOUSEWHEEL, *gui.mouse, 1 if e.y > 0 else -1)
        return keys, closed

    def show(self, canvas):
        """Displays a BGR canvas. It normally matches the window 1:1; a window smaller than
        the GUI's minimum size gets a scaled-down copy, with mouse positions scaled back."""
        w, h = self.size()
        self.scale = (canvas.shape[1] / w, canvas.shape[0] / h)
        if (canvas.shape[1], canvas.shape[0]) != (w, h):
            canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.screen.blit(self.pg.image.frombuffer(rgb, (w, h), "RGB"), (0, 0))
        self.pg.display.flip()

    def close(self):
        self.pg.display.quit()


def render_cloud(pts, cols, gui, pivot_z, w, h, f, splat=2):
    """Z-buffered point splatting on the GPU. pts (N,3) in camera coords (x right, y down,
    z forward), cols (N,3) uint8 BGR. The virtual camera orbits a pivot on the optical axis."""
    img = torch.full((h * w, 3), 18, dtype=torch.uint8, device=pts.device)
    if pts.shape[0] == 0:
        return img.view(h, w, 3)
    cy, sy, cp, sp = math.cos(gui.yaw), math.sin(gui.yaw), math.cos(gui.pitch), math.sin(gui.pitch)
    rot = pts.new_tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ pts.new_tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    p = (pts - pts.new_tensor([0.0, 0.0, pivot_z])) @ rot.T
    p = p + pts.new_tensor([gui.pan[0], gui.pan[1], 0.0])  # pan: slide the scene in the view plane
    z = p[:, 2] + pivot_z * gui.zoom
    gui.metres_per_px = float(pivot_z * gui.zoom / f)  # so a pan drag follows the pointer at the orbit centre
    front = z > 0.05
    zc = z.clamp_min(0.05)
    ui = (f * p[:, 0] / zc + w / 2).clamp(-1e6, 1e6).floor().long()
    vi = (f * p[:, 1] / zc + h / 2).clamp(-1e6, 1e6).floor().long()
    idx, zz, cc = [], [], []
    for oy in range(splat):
        for ox in range(splat):
            uu, vv = ui + ox, vi + oy
            m = front & (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            idx.append((vv * w + uu)[m])
            zz.append(z[m])
            cc.append(cols[m])
    idx, zz, cc = torch.cat(idx), torch.cat(zz), torch.cat(cc)
    zbuf = torch.full((h * w,), float("inf"), device=pts.device).scatter_reduce(0, idx, zz, reduce="amin")
    nearest = zz <= zbuf[idx]
    img[idx[nearest]] = cc[nearest]
    return img.view(h, w, 3)


def compute_normals(pts):
    """Surface normals from the point map, as a BGR uint8 image (camera-facing = blue-ish)."""
    mh, mw = pts.shape[:2]
    dx = pts[1:-1, 2:] - pts[1:-1, :-2]
    dy = pts[2:, 1:-1] - pts[:-2, 1:-1]
    n = -F.normalize(torch.linalg.cross(dx, dy), dim=-1)  # point towards the camera
    out = torch.zeros(mh, mw, 3, dtype=torch.uint8, device=pts.device)
    bgr = torch.stack((-n[..., 2], -n[..., 1], n[..., 0]), dim=-1)
    out[1:-1, 1:-1] = ((bgr * 0.5 + 0.5).clamp(0, 1) * 255).byte()
    return out


def write_ply(path, points, colours_rgb):
    """Binary PLY with float32 xyz and uint8 rgb."""
    vertex = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = colours_rgb[:, 0], colours_rgb[:, 1], colours_rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(points)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())


def label(img, text, org=(8, 20), scale=0.5):
    """White text with a black outline. The outline is drawn as shifted copies because a
    thicker stroke changes glyph spacing in newer OpenCV and would not line up."""
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), FONT, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def run(args, cam, model, dev):
    mh, mw = model.height, model.width
    K, undistort_maps = load_intrinsics(args, cam.width, cam.height)
    # Everything 3D runs at model resolution, so scale the intrinsics to it.
    fx, fy = K[0, 0] * mw / cam.width, K[1, 1] * mh / cam.height
    cx, cy = K[0, 2] * mw / cam.width, K[1, 2] * mh / cam.height
    v, u = torch.meshgrid(torch.arange(mh, device=dev), torch.arange(mw, device=dev), indexing="ij")
    # Camera coordinates: x right, y down, z forward.
    rays = torch.stack(((u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, torch.ones(mh, mw, device=dev)), dim=-1).float()
    stride_mask = torch.zeros(mh, mw, dtype=torch.bool, device=dev)
    stride_mask[:: args.stride, :: args.stride] = True
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    y_up = torch.tensor([1.0, -1.0, -1.0], device=dev)  # export convention: +Y up, camera looks down -Z

    gui = Gui(args)
    luts = {name: torch.from_numpy(lut).to(dev) for name, lut in gui.luts.items()}
    win = Window("PS3 Eye depth", gui)
    aspect = cam.width / cam.height

    vis = pcd = o3d = None
    if args.open3d:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window("Point cloud (Open3D)", 960, 720)
        vis.get_render_option().point_size = 2.0
        vis.get_render_option().background_color = np.array([0.05, 0.05, 0.07])
        pcd = o3d.geometry.PointCloud()
    cloud_added = False

    ema = None                      # temporal average of depth, for the confidence estimate
    lo, hi = gui.near, gui.far      # depth range the colours span
    pivot_z = 2.0                   # orbit centre of the cloud preview
    seq, fps, t_prev = 0, 0.0, time.perf_counter()
    while True:
        frame, seq = cam.read(seq)
        if undistort_maps is not None:
            frame = cv2.remap(frame, undistort_maps[0], undistort_maps[1], cv2.INTER_LINEAR)

        # Preprocess on GPU: BGR uint8 -> RGB float, resize, normalise.
        img = torch.from_numpy(frame).to(dev).permute(2, 0, 1).flip(0).unsqueeze(0).float().div_(255)
        rgb = F.interpolate(img, size=(mh, mw), mode="bilinear", align_corners=False)
        model.input.copy_((rgb - mean) / std)
        depth = model.infer() * args.depth_scale  # (mh, mw) metres

        # Confidence. The model does not output one, so this is a heuristic in [0, 1]:
        # low where depth jumps between neighbouring pixels (object boundaries, where
        # monocular depth smears) and where it is unstable from frame to frame.
        d4 = depth.view(1, 1, mh, mw)
        safe = depth.clamp_min(1e-3)
        rel_edge = ((F.max_pool2d(d4, 3, 1, 1) + F.max_pool2d(-d4, 3, 1, 1)).view(mh, mw)) / safe
        if ema is None:
            ema = depth.clone()
        rel_time = (depth - ema).abs() / safe
        ema = torch.lerp(ema, depth, 0.3)
        in_range = (depth > gui.near) & (depth < gui.far)
        conf = torch.exp(-(rel_edge / 0.08) ** 2) * torch.exp(-(rel_time / 0.05) ** 2) * in_range

        valid = in_range & (conf >= args.min_conf)
        if args.edge_thresh > 0:
            valid &= rel_edge < args.edge_thresh
        pts = rays * depth.unsqueeze(-1)

        # Colours span the selected range, or (auto) the 2nd-98th percentile of what is
        # inside it, smoothed over time.
        sample = depth[::4, ::4][in_range[::4, ::4]]
        if gui.auto_range and sample.numel() > 16:
            q = torch.quantile(sample, torch.tensor([0.02, 0.98], device=dev))
            lo, hi = 0.8 * lo + 0.2 * float(q[0]), 0.8 * hi + 0.2 * float(q[1])
        else:
            lo, hi = gui.near, gui.far
        level = ((depth - lo) / max(hi - lo, 1e-3)).clamp(0, 1).mul(255).long()
        if not gui.invert:
            level = 255 - level  # near = top of the colour map
        lut = luts[gui.cmap]

        # The canvas is drawn at the window's size every frame, so it is shown 1:1.
        win.set_fullscreen(gui.fullscreen)
        win_w, win_h = win.size()
        gui.resize(max(win_w, gui.min_width))
        canvas_h = max(win_h, gui.toolbar_h + gui.MIN_CONTENT + gui.STATUS)
        cells = gui.layout(canvas_h - gui.toolbar_h - gui.STATUS, aspect)
        canvas = np.zeros((canvas_h, gui.width, 3), np.uint8)
        canvas[: gui.toolbar_h] = gui.draw_toolbar()

        # Probe point: under the cursor when it is over an image view, else the centre.
        probe_pane, px, py = gui.pane_at(*gui.mouse)
        if probe_pane in (None, "cloud"):
            px = py = 0.5
        pu, pv = min(int(px * mw), mw - 1), min(int(py * mh), mh - 1)
        probe_depth, probe_conf = float(depth[pv, pu]), float(conf[pv, pu])

        depth_bgr = None
        if "depth" in cells or gui.cloud_depth_colour:
            depth_bgr = lut[level]
            depth_bgr[~in_range] = 45  # outside the selected range: flat grey
        keep = valid & stride_mask
        bgr_small = (rgb[0].permute(1, 2, 0).flip(-1) * 255).byte()
        if int(keep.sum()):
            pivot_z = 0.9 * pivot_z + 0.1 * float(depth[keep].median())

        for pane, (x0, y0, cw, ch) in cells.items():
            if pane == "camera":
                view, text = frame, "Camera"
            elif pane == "depth":
                view, text = depth_bgr.cpu().numpy(), f"Depth ({gui.cmap})"
            elif pane == "intensity":
                view, text = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), "Intensity"
            elif pane == "confidence":
                view, text = lut[conf.mul(255).long()].cpu().numpy(), "Confidence (estimated)"
            elif pane == "normals":
                view, text = compute_normals(pts).cpu().numpy(), "Normals"
            else:
                colours = depth_bgr if gui.cloud_depth_colour else bgr_small
                # Bigger splats when the view is larger than the depth map or zoomed in, to avoid holes.
                splat = min(6, max(2, math.ceil(1.6 * cw / mw / min(max(gui.zoom, 0.4), 1.0))))
                view = render_cloud(pts[keep], colours[keep], gui, pivot_z, cw, ch, fx * cw / mw, splat).cpu().numpy()
                text = "Point cloud (drag: orbit, right-drag: pan, wheel: zoom)"
            if view.shape[:2] != (ch, cw):
                view = cv2.resize(view, (cw, ch), interpolation=cv2.INTER_LINEAR)
            tile = canvas[y0:y0 + ch, x0:x0 + cw]
            tile[:] = view
            label(tile, text)
            if pane != "cloud":
                cv2.drawMarker(tile, (int(px * cw), int(py * ch)), (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
            if pane in ("depth", "confidence"):
                # Colour bar with the value range.
                top, bottom = (lo, hi) if pane == "depth" else (1.0, 0.0)
                ramp = np.linspace(0, 255, ch - 60).astype(np.int64)
                if pane == "confidence" or not gui.invert:
                    ramp = 255 - ramp
                tile[40:ch - 20, cw - 22:cw - 10] = gui.luts[gui.cmap][ramp][:, None, :]
                unit = " m" if pane == "depth" else ""
                label(tile, f"{top:.2f}{unit}", (cw - 70, 34), 0.42)
                label(tile, f"{bottom:.2f}{unit}", (cw - 70, ch - 6), 0.42)

        now = time.perf_counter()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps, t_prev = (0.9 * fps + 0.1 * inst if fps else inst), now
        where = "centre" if probe_pane in (None, "cloud") else "cursor"
        status = (f"{fps:5.1f} FPS    {where}: {probe_depth:.2f} m, confidence {probe_conf:.2f}"
                  f"    colours {lo:.2f}-{hi:.2f} m    {int(keep.sum()):,} points")
        cv2.putText(canvas, status, (8, canvas.shape[0] - 7), FONT, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        win.show(canvas)

        if vis is not None:
            pcd.points = o3d.utility.Vector3dVector((pts[keep] * y_up).double().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(rgb[0].permute(1, 2, 0)[keep].double().cpu().numpy())
            if not cloud_added and len(pcd.points):
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

        keys, closed = win.poll()
        stop = closed or "quit" in gui.pending
        for ch in keys:
            if not ch:
                continue
            if ch == "\x1b" and gui.fullscreen:
                gui.fullscreen = False
            elif ch in ("q", "\x1b"):
                stop = True
            elif ch == "f":
                gui.fullscreen = not gui.fullscreen
            elif ch in "123456":
                gui.toggle_pane(PANES[int(ch) - 1])
            elif ch == "c":
                gui.next_cmap()
            elif ch == "l":
                gui.next_layout()
            elif ch == "i":
                gui.invert = not gui.invert
            elif ch == "a":
                gui.auto_range = not gui.auto_range
            elif ch in ("left", "right", "up", "down"):
                step = {"left": (40, 0), "right": (-40, 0), "up": (0, 40), "down": (0, -40)}[ch]
                gui.pan_by(*step)
            elif ch == "r":
                gui.reset_view()
            elif ch in "+=":
                gui.zoom_by(0.88)
            elif ch == "-":
                gui.zoom_by(1 / 0.88)
            elif ch in "[]":
                gui.set_range(far=gui.far + (0.25 if ch == "]" else -0.25))
            elif ch in ";'":
                gui.set_range(near=gui.near + (0.05 if ch == "'" else -0.05))
        if stop:
            break
        if "s" in keys or "save" in gui.pending:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            np.save(f"depth_{stamp}.npy", depth.cpu().numpy())
            cv2.imwrite(f"view_{stamp}.png", canvas)
            write_ply(f"cloud_{stamp}.ply", (pts[keep] * y_up).cpu().numpy(), bgr_small[keep].flip(-1).cpu().numpy())
            print(f"Saved depth_{stamp}.npy, cloud_{stamp}.ply, view_{stamp}.png")
        gui.pending.clear()

    win.close()
    if vis is not None:
        vis.destroy_window()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="da2_metric_large.engine")
    ap.add_argument("--device", default=0, type=lambda v: int(v) if v.isdigit() else v, help="camera index or /dev/videoN")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--calib", default=None, help="calib.npz from calibrate.py")
    ap.add_argument("--hfov", type=float, default=60.0, help="horizontal FOV in degrees, used only without --calib")
    ap.add_argument("--depth-scale", type=float, default=1.0, help="multiply depth to correct metric scale")
    ap.add_argument("--min-depth", type=float, default=0.15, help="starting near limit of the range, metres")
    ap.add_argument("--max-depth", type=float, default=8.0, help="starting far limit of the range, metres")
    ap.add_argument("--range-limit", type=float, default=10.0, help="right end of the range slider, metres")
    ap.add_argument("--edge-thresh", type=float, default=0.05,
                    help="drop points where depth jumps by more than this fraction between pixels (0 = keep all)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="drop points below this estimated confidence (0-1)")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth point per axis")
    ap.add_argument("--views", default="camera,depth,cloud",
                    help="comma-separated views shown at start: " + ", ".join(PANES))
    ap.add_argument("--colormap", choices=list(COLORMAPS), default="inferno", help="starting colour map")
    ap.add_argument("--invert", action="store_true", help="map far to the top of the colour map instead of near")
    ap.add_argument("--auto-range", action="store_true", help="stretch the colour scale to the scene's depth range")
    ap.add_argument("--layout", choices=LAYOUTS, default="grid",
                    help="how views are arranged: even grid, stacked vertically, or one big view (l cycles)")
    ap.add_argument("--fullscreen", action="store_true", help="start in fullscreen (f or the toolbar button toggles)")
    ap.add_argument("--open3d", action="store_true", help="also open a separate Open3D point cloud window")
    args = ap.parse_args(argv)
    args.views = [v.strip() for v in args.views.split(",") if v.strip()]
    unknown = [v for v in args.views if v not in PANES]
    if unknown or not args.views:
        ap.error(f"--views must be a comma-separated list of: {', '.join(PANES)}")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available to PyTorch")
    model = TRTDepth(args.engine)
    cam = Camera(args.device, args.width, args.height, args.fps)
    print(f"Camera {cam.width}x{cam.height}, model input {model.width}x{model.height}")
    try:
        run(args, cam, model, torch.device("cuda"))
    finally:
        cam.close()


if __name__ == "__main__":
    main()
