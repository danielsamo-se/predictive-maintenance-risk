"""Benchmark inference time for models on CPU"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from pmrisk.models.model_versions import load_active_sequence_model


def run_benchmark(
    model: torch.nn.Module,
    x: torch.Tensor,
    iters: int,
    warmup: int,
) -> dict:
    model.eval()
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x)
    end = time.perf_counter()
    
    total_time_s = end - start
    avg_ms_per_iter = (total_time_s / iters) * 1000
    avg_ms_per_sample = avg_ms_per_iter / x.shape[0]
    
    return {
        "total_time_s": total_time_s,
        "avg_ms_per_iter": avg_ms_per_iter,
        "avg_ms_per_sample": avg_ms_per_sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sequence model inference")
    parser.add_argument("--model-name", default="seq", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--iters", type=int, default=200, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--root", default="models/production", help="Root directory")
    
    args = parser.parse_args()
    
    root = Path(args.root)
    
    print(f"Loading model '{args.model_name}' from {root}...")
    model, metadata = load_active_sequence_model(args.model_name, root=root)
    
    hparams = metadata["hparams"]
    window_l = hparams["window_l"]
    n_features = hparams["n_features"]
    
    print(f"Model: window_l={window_l}, n_features={n_features}")
    print(f"Generating dummy input: batch_size={args.batch_size}")
    
    x = torch.randn(args.batch_size, window_l, n_features)
    
    print(f"Running warmup ({args.warmup} iterations)...")
    print(f"Running benchmark ({args.iters} iterations)...")
    
    results = run_benchmark(model, x, args.iters, args.warmup)
    
    print("\n=== Benchmark Results ===")
    print(f"Batch size:           {args.batch_size}")
    print(f"Iterations:           {args.iters}")
    print(f"Total time:           {results['total_time_s']:.4f} s")
    print(f"Avg time per iter:    {results['avg_ms_per_iter']:.4f} ms")
    print(f"Avg time per sample:  {results['avg_ms_per_sample']:.4f} ms")


if __name__ == "__main__":
    main()
