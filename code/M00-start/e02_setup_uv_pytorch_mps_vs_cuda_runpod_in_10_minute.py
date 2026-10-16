"""M00E02 · env_check: is this machine ready, and how fast is it?

Prints: python + torch versions, the best device, which dtypes work on it,
and a measured matmul TFLOP/s for float32 / bfloat16 / float16.
Same file runs on the Mac (MPS) and on a RunPod box (CUDA).

Run:   .venv/bin/python code/M00-start/e02_setup_uv_pytorch_mps_vs_cuda_runpod_in_10_minute.py
Test:  .venv/bin/python -m pytest code/M00-start/e02_setup_uv_pytorch_mps_vs_cuda_runpod_in_10_minute.py -q
"""
import platform
import subprocess
import sys
import time

import torch

DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    # GPU calls are async: Python returns before the GPU finishes.
    # Without a sync, you time how fast you can *queue* work, not do it.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def dtype_works(dtype, device):
    # try a tiny matmul; some backends reject some dtypes
    try:
        a = torch.ones(8, 8, dtype=dtype, device=device)
        return bool(((a @ a)[0, 0] == 8).item())
    except (RuntimeError, TypeError):
        return False


def matmul_tflops(n, dtype, device, iters=20):
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    for _ in range(3):              # warmup: first calls pay for kernel compile/caching
        a @ b
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(device)
    secs = (time.perf_counter() - t0) / iters
    flops = 2 * n**3                # n*n outputs, each n multiplies + n adds
    return flops / secs / 1e12


def report(n=4096):
    device = pick_device()
    print(f"python   {sys.version.split()[0]}  ({platform.machine()})")
    print(f"torch    {torch.__version__}")
    print(f"device   {device}", end="")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name()}  CUDA {torch.version.cuda}")
    elif sys.platform == "darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True).stdout.strip()
        print(f"  {chip}")
    print()
    results = {}
    for dt in DTYPES:
        ok = dtype_works(dt, device)
        tf = matmul_tflops(n, dt, device) if ok else float("nan")
        results[dt] = (ok, tf)
        print(f"{str(dt):<16} supported={ok!s:<5}  matmul {n}x{n}: {tf:6.2f} TFLOP/s")
    if device.type != "cpu":
        cpu_tf = matmul_tflops(2048, torch.float32, torch.device("cpu"), iters=5)
        print(f"{'cpu float32':<16} {'':16}matmul 2048x2048: {cpu_tf:6.2f} TFLOP/s")
    return device, results


if __name__ == "__main__":
    report()


# ---------- the receipt ----------
def test_device_and_dtypes():
    device = pick_device()
    assert dtype_works(torch.float32, device)
    if device.type == "mps":
        assert dtype_works(torch.bfloat16, device)   # needs macOS 14+ and a recent torch


def test_matmul_is_correct_on_device():
    # speed means nothing if the answer is wrong: compare device matmul to CPU fp64
    device = pick_device()
    a, b = torch.randn(256, 256, dtype=torch.float64), torch.randn(256, 256, dtype=torch.float64)
    ref = a @ b
    got = (a.float().to(device) @ b.float().to(device)).cpu().double()
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-3)


def test_tflops_is_positive():
    tf = matmul_tflops(512, torch.float32, pick_device(), iters=3)
    assert tf > 0
