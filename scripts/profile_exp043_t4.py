import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.models.resnet_unet import ResNet34RegionHeadsUNet2D


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    # Avoid downloading ImageNet weights during profiling; checkpoint is loaded below.
    if cfg.get("model", {}).get("encoder_weights") == "imagenet":
        cfg["model"]["encoder_weights"] = None
    return cfg


def build_exp043_model(cfg: dict) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return ResNet34RegionHeadsUNet2D(
        n_channels=model_cfg.get("in_channels", 4),
        n_classes=model_cfg.get("num_classes", 3),
        init_features=model_cfg.get("init_features", 64),
        encoder_weights=model_cfg.get("encoder_weights", None),
    )


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def profile_macs(model: torch.nn.Module, input_shape: tuple[int, ...], device: torch.device) -> tuple[int, int]:
    macs = 0
    activation_bytes = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal macs, activation_bytes
        out = output[0] if isinstance(output, (tuple, list)) else output
        if not torch.is_tensor(out):
            return
        batch, out_channels, out_h, out_w = out.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        macs += int(batch * out_channels * out_h * out_w * kernel_ops)
        activation_bytes += int(out.numel() * out.element_size())

    def linear_hook(module, inputs, output):
        nonlocal macs, activation_bytes
        inp = inputs[0]
        out = output
        if not torch.is_tensor(out):
            return
        batch = inp.shape[0] if inp.ndim > 1 else 1
        macs += int(batch * module.in_features * module.out_features)
        activation_bytes += int(out.numel() * out.element_size())

    def activation_hook(module, inputs, output):
        nonlocal activation_bytes
        out = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(out):
            activation_bytes += int(out.numel() * out.element_size())

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU, torch.nn.Sigmoid)):
            hooks.append(module.register_forward_hook(activation_hook))

    model.eval()
    with torch.no_grad():
        _ = model(torch.randn(*input_shape, device=device))

    for hook in hooks:
        hook.remove()
    return macs, activation_bytes


def measure_latency(
    model: torch.nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float | None]:
    model.eval()
    x = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(repeats):
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            return elapsed_ms / repeats, None

        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        end = time.perf_counter()
    return (end - start) * 1000 / repeats, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp043.yaml")
    parser.add_argument("--checkpoint", default="outputs/exp043/best_model.pth")
    parser.add_argument("--output", default="cmig_exp043_t4_profile.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--slices-per-volume", type=int, default=155)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(Path(args.config))
    model = build_exp043_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    input_shape = (args.batch_size, cfg["model"].get("in_channels", 4), args.height, args.width)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    params, trainable_params = count_params(model)
    macs, activation_bytes = profile_macs(model, input_shape, device)
    latency_ms, _ = measure_latency(model, input_shape, device, args.warmup, args.repeats)
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    result = {
        "experiment": "exp043",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "input_shape": list(input_shape),
        "params": params,
        "trainable_params": trainable_params,
        "macs_per_slice": macs,
        "flops_per_slice_approx": macs * 2,
        "macs_per_volume": macs * args.slices_per_volume,
        "flops_per_volume_approx": macs * 2 * args.slices_per_volume,
        "latency_ms_per_slice_batch1": latency_ms,
        "latency_seconds_per_155_slice_volume_estimated": latency_ms * args.slices_per_volume / 1000,
        "peak_gpu_memory_mb": peak_memory_mb,
        "activation_memory_mb_hook_estimate": activation_bytes / (1024**2),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "note": (
            "Latency is forward-only with synthetic input and excludes NIfTI I/O, preprocessing, thresholding, "
            "3D reconstruction, and metric computation. Use this as model inference complexity, not full pipeline runtime."
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
