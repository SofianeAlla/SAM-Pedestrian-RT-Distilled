"""Build a desktop TensorRT engine from an ONNX model.

This is the desktop-edge runtime engine for the pedestrian expert on
the user's RTX 4070. The Orin port (cross-compile against Jetson TRT)
is out of scope for this prototype.

Tries the Python TensorRT API first; falls back to invoking `trtexec`
on PATH.

Example:
    python -m runtime.build_trt --onnx weights/best.onnx --output weights/best.engine --fp16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def build_with_python_api(
    onnx_path: Path,
    output: Path,
    fp16: bool,
    workspace_gb: int,
) -> bool:
    try:
        import tensorrt as trt  # type: ignore
    except ImportError:
        return False

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        f.write(engine)
    return True


def build_with_trtexec(
    onnx_path: Path,
    output: Path,
    fp16: bool,
    workspace_gb: int,
) -> bool:
    trtexec = shutil.which("trtexec")
    if not trtexec:
        return False
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={output}",
        f"--workspace={workspace_gb * 1024}",
    ]
    if fp16:
        cmd.append("--fp16")
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    return res.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--workspace-gb", type=int, default=4)
    args = p.parse_args()

    onnx_path = Path(args.onnx)
    output = Path(args.output)
    if not onnx_path.exists():
        raise FileNotFoundError(onnx_path)

    if build_with_python_api(onnx_path, output, args.fp16, args.workspace_gb):
        print(f"[python-api] built engine -> {output}")
        return 0
    if build_with_trtexec(onnx_path, output, args.fp16, args.workspace_gb):
        print(f"[trtexec] built engine -> {output}")
        return 0
    print(
        "Neither tensorrt Python API nor trtexec available."
        " Install TensorRT for Windows or skip the TRT step "
        "(Ultralytics can run the .pt directly for the demo)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
