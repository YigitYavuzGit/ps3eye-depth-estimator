"""Streams the viewer's frames to other programs over TCP (the DIAVision plug-in, or stream_client.py).

Deliberately simple so it is easy to read from C#: request/response, fixed little-endian headers,
raw arrays, no compression.

Request (client -> server), 12 bytes:
    char[4]  "DGET"
    uint32   flags     which blocks to send back (see below)
    uint32   last_seq  sequence number of the last frame the client got (0 = none); the server
                       waits for a newer frame, up to WAIT_SECONDS

Reply (server -> client), 52-byte header then the requested blocks in this order:
    char[4]  "DFRM"
    uint32   version   (1)
    uint32   seq       frame sequence number
    uint32   width, height
    float32  fx, fy, cx, cy   pinhole intrinsics of this pixel grid
    float32  unit_mm   millimetres per depth/cloud unit (1.0)
    uint32   flags     blocks that follow; 0 = no new frame arrived in time
    uint64   timestamp microseconds since the Unix epoch

    INTENSITY 0x1  uint8   [height][width]       grey image
    COLOUR    0x2  uint8   [height][width][3]    BGR image
    DEPTH     0x4  uint16  [height][width]       Z in millimetres, 0 = no data
    CLOUD     0x8  float32 [height][width][3]    X, Y, Z in millimetres per pixel (organised);
                                                 x right, y down, z forward from the lens;
                                                 (0, 0, 0) = no data
"""
import socket
import struct
import threading
import time

import numpy as np

DEFAULT_PORT = 5600
VERSION = 1
INTENSITY, COLOUR, DEPTH, CLOUD = 1, 2, 4, 8
REQUEST = struct.Struct("<4sII")
HEADER = struct.Struct("<4sIIII4ffIQ")
WAIT_SECONDS = 2.0


class FrameServer:
    def __init__(self, host, port):
        self.cond = threading.Condition()
        self.frame = None   # (seq, timestamp_us, intrinsics, {flag: array})
        self.clients = 0
        self.running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(4)
        self.address = self.sock.getsockname()
        threading.Thread(target=self._accept, daemon=True).start()

    def publish(self, seq, intrinsics, intensity, colour, depth_mm, cloud_mm):
        blocks = {
            INTENSITY: np.ascontiguousarray(intensity, np.uint8),
            COLOUR: np.ascontiguousarray(colour, np.uint8),
            DEPTH: np.ascontiguousarray(depth_mm, "<u2"),
            CLOUD: np.ascontiguousarray(cloud_mm, "<f4"),
        }
        with self.cond:
            self.frame = (seq, int(time.time() * 1e6), tuple(float(v) for v in intrinsics), blocks)
            self.cond.notify_all()

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
        with self.cond:
            self.cond.notify_all()

    def _accept(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    @staticmethod
    def _read(conn, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client closed")
            buf += chunk
        return bytes(buf)

    def _serve(self, conn):
        with self.cond:
            self.clients += 1
        try:
            while self.running:
                magic, flags, last_seq = REQUEST.unpack(self._read(conn, REQUEST.size))
                if magic != b"DGET":
                    return
                with self.cond:
                    self.cond.wait_for(lambda: not self.running or (self.frame and self.frame[0] != last_seq),
                                       timeout=WAIT_SECONDS)
                    frame = self.frame
                if frame is None or frame[0] == last_seq:
                    conn.sendall(HEADER.pack(b"DFRM", VERSION, last_seq, 0, 0, 0, 0, 0, 0, 1.0, 0, 0))
                    continue
                seq, stamp, intrinsics, blocks = frame
                height, width = blocks[DEPTH].shape
                flags &= INTENSITY | COLOUR | DEPTH | CLOUD
                conn.sendall(HEADER.pack(b"DFRM", VERSION, seq, width, height, *intrinsics, 1.0, flags, stamp))
                for flag in (INTENSITY, COLOUR, DEPTH, CLOUD):
                    if flags & flag:
                        conn.sendall(blocks[flag])
        except (ConnectionError, OSError, struct.error):
            pass
        finally:
            conn.close()
            with self.cond:
                self.clients -= 1
