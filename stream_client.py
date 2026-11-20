#!/usr/bin/env python3
"""Reference client for the viewer's frame stream (live_depth.py --serve).

Use it to check the stream before involving DIAVision, and as a readable description of what
the DIAVision plug-in has to do:

    python stream_client.py                 # print frame rate and value ranges
    python stream_client.py --save out      # also write the first frame as PNG / PCD
"""
import argparse
import os
import socket
import time

import numpy as np

from frame_server import CLOUD, COLOUR, DEFAULT_PORT, DEPTH, HEADER, INTENSITY, REQUEST


class StreamClient:
    def __init__(self, host="127.0.0.1", port=DEFAULT_PORT, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.seq = 0

    def _read(self, n):
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            count = self.sock.recv_into(view[got:])
            if not count:
                raise ConnectionError("server closed the connection")
            got += count
        return buf

    def get(self, flags=INTENSITY | DEPTH | CLOUD):
        """Waits for the next frame. Returns None if none arrived in time, else a dict with
        seq, width, height, fx, fy, cx, cy, timestamp and the requested arrays."""
        self.sock.sendall(REQUEST.pack(b"DGET", flags, self.seq))
        magic, _version, seq, width, height, fx, fy, cx, cy, unit_mm, got, stamp = HEADER.unpack(self._read(HEADER.size))
        if magic != b"DFRM":
            raise ConnectionError("unexpected reply")
        if not got:
            return None
        frame = dict(seq=seq, width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy, unit_mm=unit_mm, timestamp=stamp)
        if got & INTENSITY:
            frame["intensity"] = np.frombuffer(self._read(width * height), np.uint8).reshape(height, width)
        if got & COLOUR:
            frame["colour"] = np.frombuffer(self._read(width * height * 3), np.uint8).reshape(height, width, 3)
        if got & DEPTH:
            frame["depth_mm"] = np.frombuffer(self._read(width * height * 2), "<u2").reshape(height, width)
        if got & CLOUD:
            frame["cloud_mm"] = np.frombuffer(self._read(width * height * 12), "<f4").reshape(height, width, 3)
        self.seq = seq
        return frame

    def close(self):
        self.sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = until Ctrl+C)")
    ap.add_argument("--save", default=None, help="folder to write the first frame into")
    args = ap.parse_args()

    client = StreamClient(args.host, args.port)
    count, start = 0, time.perf_counter()
    try:
        while not args.frames or count < args.frames:
            frame = client.get()
            if frame is None:
                print("no new frame (is the viewer running?)")
                continue
            count += 1
            depth = frame["depth_mm"]
            if count == 1 and args.save:
                import cv2

                os.makedirs(args.save, exist_ok=True)
                cv2.imwrite(os.path.join(args.save, "image.png"), frame["intensity"])
                cv2.imwrite(os.path.join(args.save, "depth_mm.png"), depth)
                print(f"first frame written to {args.save}/")
            if count % 20 == 1:
                valid = depth[depth > 0]
                rng = f"{valid.min()}-{valid.max()} mm" if valid.size else "no valid depth"
                print(f"seq {frame['seq']}  {frame['width']}x{frame['height']}  depth {rng}  "
                      f"{count / (time.perf_counter() - start):.1f} frames/s")
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
