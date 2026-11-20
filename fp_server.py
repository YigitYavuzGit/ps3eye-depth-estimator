#!/usr/bin/env python3
"""FoundationPose bridge server for live_depth.py.

Run this INSIDE your FoundationPose environment (it imports FoundationPose's estimater.py):

    # model-based: you have a CAD mesh of the object
    python /path/to/ps3eye-depth-estimator/fp_server.py --fp-dir /path/to/FoundationPose --mesh object.obj

    # model-free: reference views of the object, in FoundationPose's ref-view format
    python /path/to/ps3eye-depth-estimator/fp_server.py --fp-dir /path/to/FoundationPose --ref-views /path/to/ob_0000001

Then start the viewer in its own environment with:  python live_depth.py --pose

The mesh must be in metres (use --mesh-scale 0.001 for a millimetre mesh).

Model-free mode trains FoundationPose's neural object field on the reference views (needs the
BundleSDF/Kaolin parts of FoundationPose installed) and caches the result as model/model.obj
inside the reference-view folder. That folder must contain rgb/, depth_enhanced/ (16-bit PNG,
millimetres), mask/, cam_in_ob/ (4x4 camera pose per view), K.txt and select_frames.yml.
"""
import argparse
import os
import sys
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np

import pose_protocol as proto


def reconstruct_from_ref_views(ref_dir, fp_dir, config):
    """Model-free path: FoundationPose's own neural object field reconstruction."""
    out_file = os.path.join(ref_dir, "model", "model.obj")
    if os.path.exists(out_file):
        print(f"Using cached reconstruction {out_file}")
        return out_file
    import yaml

    sys.path.insert(0, os.path.join(fp_dir, "bundlesdf"))
    from run_nerf import run_one_ob  # FoundationPose/bundlesdf/run_nerf.py

    with open(config or os.path.join(fp_dir, "bundlesdf", "config_ycbv.yml")) as f:
        cfg = yaml.safe_load(f)
    print("Training the neural object field from reference views (this takes a while)...")
    mesh = run_one_ob(base_dir=ref_dir, cfg=cfg)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    mesh.export(out_file)
    print(f"Reconstructed mesh written: {out_file}")
    return out_file


def build_estimator(mesh, fp_dir, debug_dir):
    import logging

    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor, dr, set_seed

    set_seed(0)
    os.makedirs(debug_dir, exist_ok=True)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                         scorer=ScorePredictor(), refiner=PoseRefinePredictor(),
                         debug_dir=debug_dir, debug=0, glctx=dr.RasterizeCudaContext())
    logging.getLogger().setLevel(logging.WARNING)  # FoundationPose logs several lines per frame
    return est


def handle(est, header, arrays, est_iter, track_iter):
    cmd = header.get("cmd")
    if cmd == "register":
        rgb, depth, K, mask = arrays
        if mask.sum() == 0:
            raise ValueError("empty object mask")
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(bool), iteration=est_iter)
    elif cmd == "track":
        rgb, depth, K = arrays
        pose = est.track_one(rgb=rgb, depth=depth, K=K, iteration=track_iter)
    else:
        raise ValueError(f"unknown command {cmd!r}")
    return np.asarray(pose, dtype=np.float64).reshape(4, 4)


def serve(est, name, extents, to_origin, host, port, est_iter=5, track_iter=2, once=False):
    """Serves one viewer at a time; a new viewer can connect after the previous one leaves."""
    with Listener((host, port)) as listener:
        print(f"FoundationPose server ready on {host}:{port} (object: {name}). Waiting for live_depth.py --pose")
        while True:
            conn = listener.accept()
            print("Viewer connected")
            try:
                proto.send(conn, {"hello": True, "name": name},
                           np.asarray(extents, np.float64), np.asarray(to_origin, np.float64))
                while True:
                    header, arrays = proto.recv(conn)
                    start = time.perf_counter()
                    try:
                        pose = handle(est, header, arrays, est_iter, track_iter)
                    except Exception as exc:  # report to the viewer and keep serving
                        traceback.print_exc()
                        proto.send(conn, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                        continue
                    proto.send(conn, {"ok": True, "ms": (time.perf_counter() - start) * 1e3}, pose)
            except (EOFError, ConnectionError, OSError):
                print("Viewer disconnected")
            finally:
                conn.close()
            if once:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp-dir", required=True, help="path to the FoundationPose repository")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mesh", help="object mesh (.obj/.ply/...), model-based mode")
    src.add_argument("--ref-views", help="reference-view folder of one object, model-free mode")
    ap.add_argument("--mesh-scale", type=float, default=1.0, help="multiply mesh units to get metres (0.001 for mm)")
    ap.add_argument("--nerf-config", default=None, help="model-free: neural field config (default: bundlesdf/config_ycbv.yml)")
    ap.add_argument("--est-refine-iter", type=int, default=5, help="refinement iterations for the first pose")
    ap.add_argument("--track-refine-iter", type=int, default=2, help="refinement iterations per tracked frame")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    ap.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    args = ap.parse_args()

    fp_dir = os.path.abspath(args.fp_dir)
    if not os.path.exists(os.path.join(fp_dir, "estimater.py")):
        raise SystemExit(f"{fp_dir} does not look like the FoundationPose repository (no estimater.py)")
    sys.path.insert(0, fp_dir)
    import trimesh

    mesh_file = args.mesh or reconstruct_from_ref_views(os.path.abspath(args.ref_views), fp_dir, args.nerf_config)
    mesh = trimesh.load(mesh_file, force="mesh")
    if args.mesh_scale != 1.0:
        mesh.apply_scale(args.mesh_scale)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    print(f"Mesh {mesh_file}: bounding box {np.round(extents, 3)} m")
    if max(extents) > 5 or max(extents) < 0.005:
        print("WARNING: that size looks wrong for metres - check --mesh-scale")

    est = build_estimator(mesh, fp_dir, os.path.join(fp_dir, "debug"))
    serve(est, os.path.basename(mesh_file), extents, to_origin, args.host, args.port,
          args.est_refine_iter, args.track_refine_iter)


if __name__ == "__main__":
    main()
