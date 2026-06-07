# tools/benchmark_model.py
# -*- coding: utf-8 -*-
"""
Model complexity analysis: parameters (M), FLOPs (G), inference time (ms).

Benchmarks all model variants: convnext, mamba, dual_naive, dual_saf, dual_rgbf, dual_full.
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dual_model import create_classifier


def count_params_m(model: nn.Module) -> float:
    """Total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_trainable_params_m(model: nn.Module) -> float:
    """Trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


@torch.no_grad()
def measure_infer_time_ms(model: nn.Module, x: torch.Tensor,
                          warmup: int = 30, iters: int = 200) -> float:
    """Mean inference latency per forward pass in milliseconds."""
    model.eval()
    for _ in range(max(0, warmup)):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(max(1, iters)):
        _ = model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) * 1000.0 / max(1, iters)


def measure_flops_g(model: nn.Module, x: torch.Tensor):
    """FLOPs in G via thop. Returns None if thop unavailable."""
    try:
        from thop import profile
        model.eval()
        macs, _ = profile(model, inputs=(x,), verbose=False)
        flops = 2.0 * float(macs)
        return flops / 1e9
    except Exception:
        return None


ALL_MODEL_NAMES = ["convnext", "mamba", "dual_naive", "dual_saf",
                    "dual_rgbf", "dual_full", "mobilevit", "davit"]


def main():
    parser = argparse.ArgumentParser(description="Model complexity benchmark")
    parser.add_argument("--model_name", type=str, default="all",
                        help="Model to benchmark, or 'all' for all variants")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="results/complexity")
    parser.add_argument("--dry_run", action="store_true",
                        help="Quick test without full iterations")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA not available, fallback to CPU.")

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

    if args.model_name == "all":
        model_names = list(ALL_MODEL_NAMES)
    else:
        model_names = [args.model_name]

    if args.dry_run:
        args.iters = min(args.iters, 5)
        args.warmup = min(args.warmup, 2)

    rows = []
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    for name in model_names:
        print(f"Benchmarking {name}...")
        error_msg = None
        try:
            model = create_classifier(name, num_classes=args.num_classes,
                                      pretrained=False).to(device)
        except Exception as e:
            error_msg = f"create_failed:{type(e).__name__}"
            print(f"  [SKIP] Failed to create model: {e}")
            rows.append({
                "model": name, "params_M": "SKIP", "trainable_M": "SKIP",
                "flops_G": "SKIP", "infer_time_ms": "SKIP",
                "gpu": gpu_name, "input_size": args.img_size,
                "error": error_msg,
            })
            continue

        params_m = count_params_m(model)
        trainable_m = count_trainable_params_m(model)

        # Inference time
        if ("mamba" in name) and device.type != "cuda":
            t_ms_str = "SKIP(cpu)"
        else:
            try:
                t_ms = measure_infer_time_ms(model, x, warmup=args.warmup, iters=args.iters)
                t_ms_str = f"{t_ms:.4f}"
            except Exception as e:
                t_ms_str = f"ERR({type(e).__name__})"
                error_msg = f"time:{type(e).__name__}"

        # FLOPs
        flops_g = measure_flops_g(model, x)
        flops_str = f"{flops_g:.3f}" if flops_g is not None else "N/A(thop)"

        rows.append({
            "model": name,
            "params_M": f"{params_m:.4f}",
            "trainable_M": f"{trainable_m:.4f}",
            "flops_G": flops_str,
            "infer_time_ms": t_ms_str,
            "gpu": gpu_name,
            "input_size": args.img_size,
            "error": error_msg or "",
        })

    # Print table
    print(f"\n{'='*90}")
    print(f"GPU: {gpu_name}")
    print(f"Input: {args.batch_size}x3x{args.img_size}x{args.img_size}")
    print(f"{'Model':<16} {'Params(M)':<12} {'Trainable(M)':<14} {'FLOPs(G)':<12} {'Infer(ms)':<12}")
    print(f"{'-'*90}")
    for r in rows:
        print(f"{r['model']:<16} {r['params_M']:<12} {r['trainable_M']:<14} "
              f"{r['flops_G']:<12} {r['infer_time_ms']:<12}")
    print(f"{'='*90}")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "complexity_results.csv")
    with open(csv_path, "w") as f:
        f.write("model,params_M,trainable_M,flops_G,infer_time_ms,gpu,input_size,error\n")
        for r in rows:
            f.write(f"{r['model']},{r['params_M']},{r['trainable_M']},"
                    f"{r['flops_G']},{r['infer_time_ms']},{r['gpu']},"
                    f"{r['input_size']},{r['error']}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
