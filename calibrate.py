#!/usr/bin/env python3
"""Checkerboard calibration for the PS3 Eye. Writes calib.npz (K, dist, size).

    python calibrate.py --cols 9 --rows 6 --square 0.025

--cols/--rows are INNER corner counts, --square is the square edge in metres.
SPACE captures a view when the board is detected, c calibrates and saves, q quits.
Take 15-25 views: different distances, tilts, and all corners of the image.
"""
import argparse
import sys

import cv2
import numpy as np


def centre_crop(frame, aspect):
    """Crops to width/height = aspect around the centre, exactly as live_depth.py does."""
    h, w = frame.shape[:2]
    if abs(w / h - aspect) < 0.02:
        return frame
    if w / h > aspect:
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return np.ascontiguousarray(frame[:, x0:x0 + new_w])
    new_h = int(round(w / aspect))
    y0 = (h - new_h) // 2
    return np.ascontiguousarray(frame[y0:y0 + new_h])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=0, type=lambda v: int(v) if v.isdigit() else v)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=0.025)
    ap.add_argument("--out", default="calib.npz")
    args = ap.parse_args()

    backend = {"linux": cv2.CAP_V4L2, "win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION}.get(sys.platform, cv2.CAP_ANY)
    cap = cv2.VideoCapture(args.device, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")

    pattern = (args.cols, args.rows)
    board = np.zeros((args.cols * args.rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_pts, img_pts, size = [], [], None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = centre_crop(frame, args.width / args.height)  # same crop as live_depth.py
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, pattern, corners, found)
        cv2.putText(view, f"views: {len(obj_pts)}  SPACE=capture  c=calibrate  q=quit", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("calibrate", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            obj_pts.append(board)
        elif key == ord("c"):
            if len(obj_pts) < 8:
                print("Need at least 8 views")
                continue
            rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
            np.savez(args.out, K=K, dist=dist, size=np.array(size))
            print(f"RMS reprojection error: {rms:.3f} px (under ~0.5 is good)")
            print(f"fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
            print(f"Saved {args.out}")
            break
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
