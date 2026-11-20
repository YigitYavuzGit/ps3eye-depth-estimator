#!/usr/bin/env python3
"""Export Depth Anything V2 (metric, indoor) to ONNX and build a TensorRT engine.

Run once per model size / input resolution:

    python build_engine.py                  # Large, best quality
    python build_engine.py --encoder base   # faster
    python build_engine.py --encoder small  # fastest (60 FPS class)

The engine is tied to this GPU and TensorRT version; rebuild after upgrading either.
"""
import argparse
import os

import torch

REPOS = {
    "small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
}


class Wrapper(torch.nn.Module):
    """float32 in, float32 depth (1, H, W) in metres out; optionally half precision inside."""

    def __init__(self, model, half):
        super().__init__()
        self.model = model.half() if half else model
        self.half = half

    def forward(self, image):
        if self.half:
            return self.model(pixel_values=image.half()).predicted_depth.float()
        return self.model(pixel_values=image).predicted_depth


def export_onnx(encoder, height, width, onnx_path, half=False):
    from transformers import AutoModelForDepthEstimation

    if height % 14 or width % 14:
        raise SystemExit("height and width must be multiples of 14 (ViT patch size)")
    model = AutoModelForDepthEstimation.from_pretrained(REPOS[encoder]).eval()
    dummy = torch.zeros(1, 3, height, width)
    kwargs = dict(
        input_names=["image"],
        output_names=["depth"],
        opset_version=17,
        do_constant_folding=True,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(Wrapper(model, half), (dummy,), onnx_path, dynamo=False, **kwargs)
        except TypeError:  # older torch without the dynamo keyword
            torch.onnx.export(Wrapper(model, half), (dummy,), onnx_path, **kwargs)
    print(f"ONNX written: {onnx_path}")


def has_fp16_flag():
    """TensorRT <= 10 picks FP16 via a builder flag. TensorRT 11 removed it: networks are
    strongly typed, so precision has to be baked into the ONNX model instead."""
    import tensorrt as trt

    return hasattr(trt.BuilderFlag, "FP16")


def build_engine(onnx_path, engine_path, fp16_flag, workspace_gb):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 0
    explicit = getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", None)
    if explicit is not None:  # required on TensorRT 8, implicit on 10
        flags = 1 << int(explicit)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    # parse_from_file resolves external weight files next to the .onnx
    if not parser.parse_from_file(os.path.abspath(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16_flag:
        config.set_flag(trt.BuilderFlag.FP16)

    print("Building TensorRT engine (takes a few minutes)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("Engine build failed")
    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine written: {engine_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=REPOS, default="large")
    ap.add_argument("--height", type=int, default=392, help="model input height (multiple of 14)")
    ap.add_argument("--width", type=int, default=518, help="model input width (multiple of 14)")
    ap.add_argument("--fp32", action="store_true", help="disable FP16 (slower; use if depth looks broken)")
    ap.add_argument("--workspace-gb", type=int, default=6)
    ap.add_argument("--onnx-only", action="store_true", help="export ONNX and stop")
    ap.add_argument("--out", default=None, help="engine path (default: da2_metric_<encoder>.engine)")
    args = ap.parse_args()

    fp16 = not args.fp32
    # Old TensorRT: FP32 ONNX + FP16 builder flag. TensorRT 11+: export the ONNX in FP16.
    use_flag = fp16 and (args.onnx_only or has_fp16_flag())
    half_onnx = fp16 and not use_flag
    stem = f"da2_metric_{args.encoder}_{args.height}x{args.width}"
    onnx_path = stem + ("_fp16" if half_onnx else "") + ".onnx"
    engine_path = args.out or f"da2_metric_{args.encoder}.engine"

    if not os.path.exists(onnx_path):
        export_onnx(args.encoder, args.height, args.width, onnx_path, half=half_onnx)
    else:
        print(f"Reusing {onnx_path}")
    if not args.onnx_only:
        build_engine(onnx_path, engine_path, use_flag, args.workspace_gb)


if __name__ == "__main__":
    main()
