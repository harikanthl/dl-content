# EP002 · M00E02 · Setup: uv, PyTorch, MPS vs CUDA, RunPod in 10 minutes

**Module:** Start Here · **Hardware:** Mac (+ RunPod for the CUDA half) · **Target length:** 10-13 min
**Code:** `code/M00-start/e02_setup_uv_pytorch_mps_vs_cuda_runpod_in_10_minute.py` (the `env_check`) · **Explainer:** `explainers/scenes/m00e02_setup_uv_pytorch_mps_vs_cuda_run.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "My first GPU benchmark said this Mac does 2,300 teraflops. That's more than an H100. It's also
> a lie, and the bug is one missing line. Let's set up the environment and measure it properly."

## ❓ The question (why this matters, 45 s)
Every result in this series is a number: a loss, a speed, a memory figure. If your environment
is not reproducible, nobody can check your numbers, including you in three weeks.

Two machines. A Mac for the first 62 episodes (M00 to M05), and rented NVIDIA GPUs for the parts
that need CUDA: Triton kernels, FlashAttention, multi-GPU training. Same code, same lockfile, one
script that tells you what you're running on and how fast it is.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: the GPU is a restaurant kitchen, Python is the waiter.** When you call `a @ b` on the GPU,
> the waiter writes the order on a ticket and walks away. He doesn't wait for the food. If you time
> how long the waiter takes, you're timing handwriting, not cooking. `torch.mps.synchronize()` is
> the waiter standing at the pass until the plate comes out.

**The second picture: uv is a recipe card with exact quantities.** `pyproject.toml` says "torch,
recent." `uv.lock` says "torch 2.14.1, this exact file, this hash." Anyone with the lockfile cooks
the same dish.

**MPS vs CUDA, one line each:**
- **MPS** = Metal Performance Shaders. PyTorch's backend for the Apple GPU. Unified memory, so the
  GPU shares the Mac's RAM (16 GB here). No float64. Some ops missing.
- **CUDA** = NVIDIA. Where every serious LLM is trained. Tensor cores, bf16 at hundreds of TFLOP/s,
  Triton, FlashAttention, NCCL for multi-GPU.

## 📐 The math (1 min)
Matmul FLOPs. An `(n, n) @ (n, n)` product has `n * n` outputs. Each output is a dot product of
length `n`: `n` multiplies and `n` adds. So:

```
FLOPs = 2 * n^3
TFLOP/s = 2 * n^3 / seconds / 1e12
```
For `n = 4096`: `2 * 4096^3 = 1.37e11` FLOPs per matmul. At 3.4 TFLOP/s that's about 40 ms.

## 💻 Code it (5 min)

**Part 1: the env (type in terminal, Mac)**
```bash
brew install uv                  # or: curl -LsSf https://astral.sh/uv/install.sh | sh
uv init dl-content && cd dl-content
uv python pin 3.12
uv add torch numpy einops matplotlib pytest
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
git add pyproject.toml uv.lock   # the lockfile IS the reproducibility. commit it.
```

**Part 2: env_check (the key lines)**
```python
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync(device):
    # GPU calls are async: Python returns before the GPU finishes.
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

def matmul_tflops(n, dtype, device, iters=20):
    a = torch.randn(n, n, dtype=dtype, device=device)
    b = torch.randn(n, n, dtype=dtype, device=device)
    for _ in range(3):              # warmup: first calls pay for kernel setup
        a @ b
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    sync(device)
    secs = (time.perf_counter() - t0) / iters
    return 2 * n**3 / secs / 1e12
```
Plus `dtype_works()`: try an 8x8 matmul in each dtype, catch the error if the backend refuses it.

**Part 3: the same file on RunPod (CUDA)**
1. Add credit first. RunPod bills from a prepaid balance. **Leave auto top-up off.** That balance is
   your spend cap: when it runs out, pods stop. Put in $10, not $100.
2. Deploy a pod from a PyTorch template (RTX 4090 for most of the series).
3. In the pod terminal: `pip install uv && git clone <repo> && cd <repo> && uv sync && uv run python code/M00-start/e02_*.py`
4. **Terminate** the pod when done, not just stop. A stopped pod still bills for its volume disk.

On Linux x86, the default PyPI torch wheel ships with CUDA, so the same `uv.lock` workflow works.
If `torch.cuda.is_available()` is False on a GPU pod, check `nvidia-smi` first, then the torch build
(`torch.version.cuda` is None on a CPU-only wheel).

## ✅ The receipt (1 min)
Speed means nothing if the answer is wrong. The test runs a 256x256 matmul on the device in float32
and compares it to the CPU in float64:
```python
torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-3)
```
Also checks bf16 works on MPS. `pytest ... -q` → **3 passed** (measured on M4 Mac mini).

## 📊 Measure (1 min)
Measured on M4 Mac mini (10-core GPU, 16 GB), torch 2.14.1:
```
python   3.12.13  (arm64)
torch    2.14.1
device   mps  Apple M4

torch.float32    supported=True   matmul 4096x4096:   3.39 TFLOP/s
torch.bfloat16   supported=True   matmul 4096x4096:   3.78 TFLOP/s
torch.float16    supported=True   matmul 4096x4096:   3.78 TFLOP/s
cpu float32                      matmul 2048x2048:   1.90 TFLOP/s
```

| What | Measured on M4 Mac mini |
|---|---|
| MPS fp32, n=512 / 1024 / 2048 / 4096 / 8192 | 1.09 / 1.67 / 3.37 / 3.39 / 3.07 TFLOP/s |
| MPS bf16, n=4096 | 3.78 TFLOP/s (only ~10% faster than fp32) |
| CPU fp32, n=4096 (Accelerate) | 1.86 TFLOP/s |
| CPU **bf16**, n=1024 | **0.003** TFLOP/s (about 500x slower than CPU fp32) |
| MPS fp32 **without sync** | **2,309 "TFLOP/s"** (fake) |

Reference points to say, not measured here (vendor specs, dense): RTX 4090 about 82 TFLOP/s fp32
and about 165 TFLOP/s bf16 on tensor cores. H100 SXM about 989 TFLOP/s bf16. So the 4090 is ~40x
this Mac on bf16 matmul, and the H100 ~250x. That's why M07 onward rents GPUs.

**Talking point:** on the Mac, bf16 barely beats fp32. On NVIDIA it's 2x or more, because tensor
cores. Don't expect Mac speedups from lower precision.

## ⚠️ Gotchas
- **No sync = fake numbers.** The 2,309 TFLOP/s line. Every benchmark in this series syncs before
  stopping the clock.
- **No warmup = slow numbers.** The first call on MPS or CUDA pays for kernel compilation and memory
  allocation. Throw away the first few runs.
- **Small matrices lie low.** n=512 gets 1.09 TFLOP/s, n=4096 gets 3.39. Small kernels are
  dominated by launch overhead. Benchmark at the size you'll use.
- **No float64 on MPS.** `torch.zeros(2, dtype=torch.float64, device="mps")` →
  `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.`
  Finite-difference gradient checks run on CPU.
- **Missing MPS ops:** `NotImplementedError: The operator 'aten::...' is not currently implemented
  for the MPS device`. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` to fall back to CPU for that op. Know
  that it's slow.
- **bf16 on CPU is a trap** (0.003 TFLOP/s above). Use bf16 on the GPU only.
- **RunPod "stop" is not "terminate".** Stopped pods keep billing for volume storage.
- **pytest file naming.** Our files are named `e02_*.py`, not `test_*.py`, so run pytest on the file
  path directly. `pytest code/M00-start/` with no file finds nothing.

## 🎤 Interview lines
- *"GPU ops are asynchronous, so I always synchronize before reading the timer, and I warm up
  first. Otherwise I'm measuring kernel launch, not compute."*
- *"Matmul is 2n³ FLOPs. I report achieved TFLOP/s against the hardware peak, because that tells me
  how much headroom is left."*
- *"I pin the environment with a lockfile so results are reproducible across my laptop and a cloud
  GPU."*

## 🔗 Next
Next is **M01E01: Tensors, what they actually are.** We just multiplied two 4096x4096 tensors. Next
we open one up and find out it's a flat list of numbers with a recipe attached.

## 📹 Metadata
- **YouTube title:** `PyTorch setup on a Mac (MPS) and a rented GPU, plus a real TFLOPs benchmark`
- **Thumbnail (3-5 words):** `2,300 TFLOPS? NO.`
- **Short (which beat, 45-60 s):** the fake benchmark. Run without sync: 2,309 TFLOP/s. Add one line: 3.39. Caption: "Your GPU benchmark is lying."
- **X post:** Benchmarked matmul on my M4 Mac mini. Without torch.mps.synchronize() it reports 2,309 TFLOP/s. With it: 3.39 fp32, 3.78 bf16. GPU calls are async; time the cooking, not the order ticket. Code + video:
- **Sources:** [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) · [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html) · [RunPod docs](https://docs.runpod.io) · [uv docs](https://docs.astral.sh/uv/)
