#!/usr/bin/env python3
"""Convert a PyTorch .pt checkpoint to ONNX.

Supports:
- Full serialized nn.Module checkpoints
- Dict checkpoints with state dicts (e.g. keys: model_state/state_dict)
- ResNet-style state-dict checkpoints with configurable/inferred
  input channels and output classes
"""

from __future__ import annotations

import argparse
import pathlib
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .pt to .onnx")
    parser.add_argument("--pt", required=True, type=Path, help="Input .pt checkpoint path")
    parser.add_argument("--onnx", required=True, type=Path, help="Output .onnx path")

    parser.add_argument(
        "--arch",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture to build for state-dict checkpoints",
    )
    parser.add_argument(
        "--state-dict-key",
        default="auto",
        help="Checkpoint key containing state_dict. Use 'auto' to detect common keys.",
    )
    parser.add_argument("--num-classes", type=int, default=None, help="Output classes")
    parser.add_argument("--in-channels", type=int, default=None, help="Input channels")

    parser.add_argument("--height", type=int, default=224, help="Input height")
    parser.add_argument("--width", type=int, default=224, help="Input width")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy export batch size")

    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--use-dynamo",
        action="store_true",
        help="Use the new ONNX exporter (requires onnxscript in many torch versions).",
    )
    parser.add_argument(
        "--dynamic-spatial",
        action="store_true",
        help="Export with dynamic height/width axes in addition to dynamic batch",
    )
    return parser.parse_args()


def extract_state_dict(ckpt: Any, key: str) -> dict[str, torch.Tensor] | None:
    if isinstance(ckpt, nn.Module):
        return None
    if not isinstance(ckpt, dict):
        return None

    if key != "auto":
        candidate = ckpt.get(key)
        if isinstance(candidate, dict):
            return candidate
        raise ValueError(f"Checkpoint key '{key}' not found or not a dict")

    if "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        return ckpt["model_state"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]

    # Some checkpoints are plain state_dict dicts.
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    return None


def infer_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int | None, int | None, int | None]:
    in_channels = None
    num_classes = None
    fc_in = None

    conv1_w = state_dict.get("conv1.weight")
    if isinstance(conv1_w, torch.Tensor) and conv1_w.ndim >= 2:
        in_channels = int(conv1_w.shape[1])

    fc_w = state_dict.get("fc.weight")
    if isinstance(fc_w, torch.Tensor) and fc_w.ndim == 2:
        num_classes = int(fc_w.shape[0])
        fc_in = int(fc_w.shape[1])

    return in_channels, num_classes, fc_in


def build_resnet(arch: str, in_channels: int, num_classes: int, fc_in_override: int | None) -> nn.Module:
    from torchvision import models

    constructors = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
    }
    model = constructors[arch](weights=None)

    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels,
            model.conv1.out_channels,
            kernel_size=model.conv1.kernel_size,
            stride=model.conv1.stride,
            padding=model.conv1.padding,
            bias=False,
        )

    fc_in = fc_in_override if fc_in_override is not None else model.fc.in_features
    model.fc = nn.Linear(fc_in, num_classes)
    return model


def _torch_load_with_windows_path_compat(path: Path, *, weights_only: bool | None) -> Any:
    """Load checkpoints saved on Windows that contain pathlib.WindowsPath objects."""
    original_windows_path = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    try:
        kwargs = {"map_location": "cpu"}
        if weights_only is not None:
            kwargs["weights_only"] = weights_only
        return torch.load(path, **kwargs)
    finally:
        pathlib.WindowsPath = original_windows_path


def torch_load_compat(path: Path) -> Any:
    """Robust torch.load across torch versions and cross-platform path pickles."""
    # Try safest mode first (often enough for state-dict checkpoints).
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Common with checkpoints that include non-tensor metadata (e.g. pathlib objects).
        # Fall back to trusted full unpickling below.
        pass
    except TypeError:
        # Older torch without weights_only argument.
        pass
    except NotImplementedError as exc:
        if "WindowsPath" in str(exc):
            try:
                return _torch_load_with_windows_path_compat(path, weights_only=True)
            except TypeError:
                return _torch_load_with_windows_path_compat(path, weights_only=None)
        raise

    # Fall back to standard pickle mode for full module checkpoints.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except pickle.UnpicklingError:
        # Some torch builds may still route this through unpickling errors.
        return _torch_load_with_windows_path_compat(path, weights_only=False)
    except NotImplementedError as exc:
        if "WindowsPath" in str(exc):
            return _torch_load_with_windows_path_compat(path, weights_only=False)
        raise
    except TypeError:
        # Older torch without weights_only argument.
        try:
            return torch.load(path, map_location="cpu")
        except NotImplementedError as exc:
            if "WindowsPath" in str(exc):
                return _torch_load_with_windows_path_compat(path, weights_only=None)
            raise


def resolve_model(args: argparse.Namespace) -> tuple[nn.Module, int]:
    ckpt = torch_load_compat(args.pt)

    if isinstance(ckpt, nn.Module):
        model = ckpt.eval()
        if args.in_channels is None:
            conv1 = getattr(model, "conv1", None)
            if isinstance(conv1, nn.Conv2d):
                args.in_channels = int(conv1.in_channels)
            else:
                args.in_channels = 3
        return model, args.in_channels

    state_dict = extract_state_dict(ckpt, args.state_dict_key)
    if state_dict is None:
        raise ValueError(
            "Unsupported checkpoint format. Provide a full nn.Module checkpoint or "
            "a dict containing a state dict (model_state/state_dict)."
        )

    inferred_in, inferred_classes, inferred_fc_in = infer_dims_from_state_dict(state_dict)
    in_channels = args.in_channels if args.in_channels is not None else (inferred_in or 3)
    num_classes = args.num_classes if args.num_classes is not None else inferred_classes
    if num_classes is None:
        raise ValueError("Could not infer num_classes from state_dict; pass --num-classes")

    model = build_resnet(args.arch, in_channels, num_classes, inferred_fc_in)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, in_channels


def export_onnx(
    model: nn.Module,
    onnx_path: Path,
    in_channels: int,
    batch_size: int,
    height: int,
    width: int,
    opset: int,
    use_dynamo: bool,
    dynamic_spatial: bool,
) -> None:
    dummy = torch.randn(batch_size, in_channels, height, width, device="cpu")

    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    if dynamic_spatial:
        dynamic_axes["input"][2] = "height"
        dynamic_axes["input"][3] = "width"

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=use_dynamo,
    )
    try:
        torch.onnx.export(model, dummy, onnx_path.as_posix(), **export_kwargs)
    except TypeError as exc:
        # Older torch versions may not accept 'dynamo' kwarg.
        if "dynamo" in str(exc):
            export_kwargs.pop("dynamo", None)
            torch.onnx.export(model, dummy, onnx_path.as_posix(), **export_kwargs)
        else:
            raise
    except ModuleNotFoundError as exc:
        if exc.name == "onnxscript":
            raise ModuleNotFoundError(
                "Missing dependency 'onnxscript' for dynamo ONNX export. "
                "Either rerun without --use-dynamo (legacy exporter) "
                "or install dependencies: pip install onnx onnxscript"
            ) from exc
        raise


def main() -> None:
    args = parse_args()
    if not args.pt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.pt}")

    model, in_channels = resolve_model(args)
    export_onnx(
        model=model,
        onnx_path=args.onnx,
        in_channels=in_channels,
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        opset=args.opset,
        use_dynamo=args.use_dynamo,
        dynamic_spatial=args.dynamic_spatial,
    )
    print(f"Exported ONNX: {args.onnx}")


if __name__ == "__main__":
    main()
