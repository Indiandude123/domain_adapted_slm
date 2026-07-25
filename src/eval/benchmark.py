"""Inference latency/throughput and memory-footprint benchmarking (4-bit vs fp16)."""
from __future__ import annotations

import time

import torch


def measure_peak_memory_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)


def benchmark_inference(model, tokenizer, texts: list[str], max_length: int = 2048, n_warmup: int = 2) -> dict:
    device = next(model.parameters()).device
    inputs = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**inputs)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "batch_size": len(texts),
        "latency_s": elapsed,
        "throughput_examples_per_s": len(texts) / elapsed,
        "peak_memory_mb": measure_peak_memory_mb(),
    }
