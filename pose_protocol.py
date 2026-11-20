"""Wire format shared by live_depth.py and fp_server.py.

The two run in different Python environments, so nothing is pickled (numpy pickles are not
portable across numpy versions). A message is a JSON header followed by raw array buffers,
sent over a multiprocessing.connection, which handles the framing. Standard library + numpy only.

viewer -> server
    {"cmd": "register"}  arrays: rgb uint8 (H,W,3), depth float32 (H,W) metres with 0 = invalid,
                                 K float64 (3,3), mask uint8 (H,W)
    {"cmd": "track"}     arrays: rgb, depth, K
server -> viewer
    on connect: {"hello": true, "name": ...}  arrays: extents (3,), to_origin (4,4)
    per request: {"ok": true, "ms": ...}      arrays: pose float64 (4,4), object in camera
                 {"ok": false, "error": ...}
"""
import json

import numpy as np

DEFAULT_PORT = 5599


def send(conn, header, *arrays):
    arrays = [np.ascontiguousarray(a) for a in arrays]
    header = dict(header, arrays=[{"dtype": str(a.dtype), "shape": list(a.shape)} for a in arrays])
    conn.send_bytes(json.dumps(header).encode("utf-8"))
    for a in arrays:
        conn.send_bytes(a.tobytes())


def recv(conn):
    header = json.loads(conn.recv_bytes().decode("utf-8"))
    arrays = []
    for spec in header.pop("arrays", []):
        data = np.frombuffer(conn.recv_bytes(), dtype=np.dtype(spec["dtype"]))
        arrays.append(data.reshape(spec["shape"]).copy())  # copy: writable, owns its memory
    return header, arrays
