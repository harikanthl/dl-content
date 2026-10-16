"""M01E12 · MNIST MLP end to end on the M4 GPU

Raw idx files -> tensors -> MLP -> >97% test accuracy. Then CPU vs MPS, honestly timed.
Data: the 4 MNIST .gz files in DL-content/data/mnist/ (no torchvision needed).
Run:   .venv/bin/python code/M01-pytorch/e12_mnist_mlp_end_to_end_on_the_m4_gpu.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e12_mnist_mlp_end_to_end_on_the_m4_gpu.py -q
       (SKIP_MNIST=1 skips the training test)
"""
import gzip
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[2] / "data" / "mnist"
URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # curl these 4 files into DATA


# ---------------------------------------------------------------------------
# 1. Data. The idx format: a big-endian header, then raw uint8 bytes.
#    images: magic, n, rows, cols (16 bytes) | labels: magic, n (8 bytes)
# ---------------------------------------------------------------------------
def read_idx(name, header_bytes):
    with gzip.open(DATA / f"{name}.gz", "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=header_bytes)


def load_mnist(split):  # split: "train" or "t10k"
    x = read_idx(f"{split}-images-idx3-ubyte", 16).reshape(-1, 28 * 28)  # (N, 784) uint8
    y = read_idx(f"{split}-labels-idx1-ubyte", 8)  # (N,)
    x = torch.from_numpy(x.copy()).float().div_(255)  # .copy(): numpy buffer is read-only
    x = (x - 0.1307) / 0.3081  # standardize with the train-set mean/std
    return x, torch.from_numpy(y.copy()).long()


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else
                        "cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 2. Model: 784 -> 512 -> 256 -> 10. About 535k parameters.
# ---------------------------------------------------------------------------
def make_mlp():
    return nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(256, 10),  # raw logits: cross_entropy does the softmax
    )


def sync(device):  # GPU work is async: without this, your timer measures nothing
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def accuracy(model, x, y, batch_size=2000):
    model.eval()
    correct = sum((model(x[i:i + batch_size]).argmax(-1) == y[i:i + batch_size]).sum().item()
                  for i in range(0, len(x), batch_size))
    return correct / len(x)


def fit(epochs=3, batch_size=128, lr=1e-3, device=None, seed=0, verbose=True):
    torch.manual_seed(seed)
    device = device or get_device()
    xtr, ytr = load_mnist("train")
    xte, yte = load_mnist("t10k")
    # MNIST is 188 MB as float32. Fits on the GPU, so move it ONCE, not per batch.
    xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))
    model = make_mlp().to(device)  # model and data must live on the same device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    times = []
    for epoch in range(epochs):
        model.train()
        sync(device)
        t0 = time.perf_counter()
        perm = torch.randperm(len(xtr), device=device)  # shuffle on the device
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sync(device)
        times.append(time.perf_counter() - t0)
        acc = accuracy(model, xte, yte)
        if verbose:
            print(f"[{device.type}] epoch {epoch} | last-batch loss {loss.item():.4f} | "
                  f"test acc {acc:.4f} | {times[-1]:.2f}s")
    return acc, times


if __name__ == "__main__":
    n_params = sum(p.numel() for p in make_mlp().parameters())
    print(f"params: {n_params:,}")
    for dev in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        for bs in (128, 1024):
            acc, times = fit(epochs=3, batch_size=bs, device=torch.device(dev), verbose=False)
            # epoch 0 includes warm-up (MPS compiles kernels on first use), so report the rest
            print(f"{dev:4s} bs={bs:5d} | test acc {acc:.4f} | "
                  f"epoch0 {times[0]:.2f}s | steady {np.mean(times[1:]):.2f}s/epoch")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
HAVE_DATA = all((DATA / f"{n}.gz").exists() for n in
                ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"])
needs_data = pytest.mark.skipif(os.environ.get("SKIP_MNIST") == "1" or not HAVE_DATA,
                                reason=f"MNIST not in {DATA} (download from {URL}) or SKIP_MNIST=1")


@needs_data
def test_idx_parsing():
    x, y = load_mnist("t10k")
    assert x.shape == (10000, 784) and y.shape == (10000,)
    assert y.min() == 0 and y.max() == 9
    assert abs(x.mean().item()) < 0.1  # standardized (test set, so not exactly 0)


@needs_data
def test_mnist_mlp_beats_97_percent():
    acc, _ = fit(epochs=3, batch_size=128, verbose=False)  # on MPS if available, ~1.5 s
    assert acc > 0.97, acc


def test_mlp_shapes_any_device():
    dev = get_device()
    out = make_mlp().to(dev)(torch.randn(5, 784, device=dev))
    assert out.shape == (5, 10) and out.device.type == dev.type
