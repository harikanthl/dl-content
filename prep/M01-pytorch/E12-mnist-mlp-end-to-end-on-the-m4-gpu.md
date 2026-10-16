# EP014 · M01E12 · MNIST MLP end to end on the M4 GPU

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e12_mnist_mlp_end_to_end_on_the_m4_gpu.py` · **Explainer:** `explainers/scenes/m01e12_mnist_mlp_end_to_end_on_the_m4_g.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "97.7% on MNIST in under two seconds of training. And here's the part nobody tells you:
> on this Mac, the CPU almost kept up with the GPU. Let's find out when the GPU actually wins."

## ❓ The question (why this matters, 45 s)
Seven episodes of parts. Today they run as one machine: raw bytes off disk, a Module, a
loss, an optimizer, the loop, and a device. MNIST is the "hello world", but the decisions
are the real ones: where does the data live, where does the model live, how do you time a
GPU without lying to yourself.

Every model in this series runs on this Mac first and on a rented H100 second. Code that
says `.to(device)` everywhere and never hardcodes `"cuda"` or `"mps"` moves between them
with zero edits. That habit starts today.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: the GPU is a kitchen across the street.**
> It has 10 ovens. Your apartment (the CPU) has 1 really good oven. To use the big kitchen
> you have to carry ingredients across the street (copy to device), and every dish you
> order has a fixed walk there and back (kernel launch overhead).
>
> Baking one cupcake at a time? Your apartment wins. You spend all your time walking.
> Baking 1,000 cupcakes in one trip? The big kitchen crushes it.
>
> So: carry the ingredients over **once** (put the whole dataset on the GPU), and order in
> **big batches**. Both of today's speedups are exactly that.

The picture: a timeline. Top row CPU: one long block per batch. Bottom row MPS: tiny
compute blocks with gaps between them (launch overhead). Make the batch bigger and the
gaps shrink relative to the blocks.

## 🪤 The trap
**Timing a GPU without synchronizing.** GPU calls are asynchronous: `y = x @ x` on MPS
returns as soon as the work is *queued*. Measured on M4 Mac mini, 20 matmuls of
`4096 x 4096`:
```
no sync    4.4 ms
with sync  816.8 ms
```
A 185x lie. Every benchmark in this series calls `torch.mps.synchronize()` (or
`torch.cuda.synchronize()`) before reading the clock. The `sync()` helper in the code does it.

**Second trap: model and data on different devices.**
`RuntimeError: Tensor for argument input is on cpu but expected on mps` (measured, torch 2.14).
Loud, at least. The fix is a rule: one `device` variable, created once, and everything
goes `.to(device)`.

## 📐 The math (derive on screen, 2–3 min)
1. **The data.** idx format: big-endian header then raw bytes. Images: 4 int32s (magic,
   N, rows, cols) = 16 bytes. Labels: 2 int32s = 8 bytes. So
   `np.frombuffer(bytes, uint8, offset=16).reshape(-1, 784)`. That's the whole parser.
2. **Normalize.** `x = (x/255 − 0.1307) / 0.3081`. Those are MNIST's train-set pixel mean
   and std. Zero-mean inputs keep the first layer's pre-activations centered, so ReLUs
   aren't all on or all off at init.
3. **Model.** `784 → 512 → 256 → 10`, ReLU, Dropout(0.1).
   Params: `784·512+512 + 512·256+256 + 256·10+10 = 401,920 + 131,328 + 2,570 = 535,818`.
4. **Memory.** Train set as float32: `60,000 · 784 · 4 B = 188 MB`. The M4 has unified
   memory, so "moving to the GPU" is cheap, and 188 MB is nothing. Move it once.
5. **Shapes per step.** `x (128, 784)`, logits `(128, 10)`, `y (128,)`.
   Steps per epoch: `⌈60000/128⌉ = 469`.

## 💻 Code it (from scratch, 4–5 min)
```python
def read_idx(name, header_bytes):
    with gzip.open(DATA / f"{name}.gz", "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=header_bytes)

def load_mnist(split):                               # "train" or "t10k"
    x = read_idx(f"{split}-images-idx3-ubyte", 16).reshape(-1, 784)
    y = read_idx(f"{split}-labels-idx1-ubyte", 8)
    x = torch.from_numpy(x.copy()).float().div_(255)  # .copy(): frombuffer is read-only
    x = (x - 0.1307) / 0.3081
    return x, torch.from_numpy(y.copy()).long()

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))   # once
model = make_mlp().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(epochs):
    model.train()
    perm = torch.randperm(len(xtr), device=device)   # shuffle where the data lives
    for i in range(0, len(xtr), batch_size):
        idx = perm[i:i + batch_size]
        loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    acc = accuracy(model, xte, yte)                  # eval() + no_grad inside
```
No torchvision. Download the 4 `.gz` files (11 MB total) from
`https://ossci-datasets.s3.amazonaws.com/mnist/` into `data/mnist/`. Show the `curl`.

Why no DataLoader here, one episode after we built one: the whole dataset is already a
tensor on the device. Slicing it with a random permutation *is* a DataLoader, minus
per-example Python calls. Measured below, it's 2x faster on MPS.

## ✅ The receipt (1 min)
- `test_mnist_mlp_beats_97_percent`: 3 epochs, batch 128, Adam 1e-3, on MPS if present,
  asserts test accuracy > 0.97. Takes about 1.5 s. Skips cleanly with `SKIP_MNIST=1` or
  if the data folder is empty.
- `test_idx_parsing`: shapes `(10000, 784)` and `(10000,)`, labels 0..9, standardized.
- `test_mlp_shapes_any_device`: logits `(5, 10)` on whatever device you have.

`pytest -q` → **3 passed** in about 2.4 s.

## 📊 Measure (1 min)
All measured on M4 Mac mini, torch 2.14, 3 epochs, steady-state = mean of epochs 2–3:

| device | batch | test acc | epoch 1 | steady s/epoch |
|---|---|---|---|---|
| CPU | 128 | 0.9772 | 0.54 s | **0.52 s** |
| MPS | 128 | 0.9762 | 0.48 s | **0.34 s** |
| CPU | 1024 | 0.9686 | 0.27 s | **0.24 s** |
| MPS | 1024 | 0.9681 | 0.56 s | **0.10 s** |

- MPS is **1.5x** faster at batch 128 and **2.4x** at batch 1024. The cupcake analogy, in numbers.
- The M4 CPU is fast here because PyTorch's macOS build does CPU matmuls through Apple's
  Accelerate, which uses the chip's matrix units (`torch.__config__.show()` prints
  `BLAS_INFO=accelerate`). Don't assume "GPU = 10x". Measure.
- Bigger batch is faster per epoch but **less accurate after 3 epochs** (0.968 vs 0.977):
  8x fewer optimizer steps. Speed per epoch is not speed to a target accuracy.
- **DataLoader tax:** same model with `DataLoader(TensorDataset(x, y), batch_size=128,
  shuffle=True)`, copying each batch to the device: CPU 0.64 s/epoch, **MPS 0.71 s/epoch**.
  On-device slicing: 0.34 s. Twice as fast, by not calling `__getitem__` 60,000 times.
- Robustness: 2 epochs on MPS across 4 seeds: 0.9739, 0.9724, 0.9753, 0.9760. That's why
  the test uses 3 epochs: 97% with margin. 10 epochs peaks at 0.981. An MLP tops out
  around 98%; CNNs get 99%+.

## ⚠️ Gotchas
- **No sync before the timer** → MPS looks 100x faster than it is.
- **Timing epoch 1 on MPS.** It includes kernel compilation. Report steady state.
- **`torch.from_numpy` on a `frombuffer` array** → `UserWarning: The given NumPy array is
  not writable`. `.copy()` first.
- **float64 on MPS** → `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
  framework doesn't support float64.` NumPy defaults to float64; `.float()` before `.to("mps")`.
- **Calling `.item()` every step.** Each `.item()` forces a sync. We log once per epoch.
  In M06 this matters a lot.
- **Labels as uint8.** `cross_entropy` wants int64. `.long()`.
- **Forgetting to normalize.** Still trains, a bit slower and less stable. Not a crash, a tax.
- **Reading MPS accuracy to the 4th decimal.** MPS kernels can reorder float sums run to
  run. CPU 0.9772 vs MPS 0.9762 is noise, not a quality difference.

## 🎤 Interview lines
- *"GPU ops are asynchronous, so I synchronize before stopping the timer; otherwise I'm
  timing the kernel launch, not the kernel."*
- *"For a small dataset I keep the whole thing resident on the device and shuffle with a
  device-side permutation. It removes per-sample Python overhead, which on MNIST is half
  the epoch."*
- *"Small models are launch-overhead bound on GPU. Larger batches amortize the overhead,
  but fewer steps per epoch can cost accuracy at a fixed epoch budget."*
- *"I write device-agnostic code: one device variable, everything .to(device), so the same
  script runs on MPS, CUDA or CPU."*

## 🔗 Next
That run took two seconds. A real one takes two weeks, and it will crash. Next, M01E13:
checkpoints that include the optimizer and the RNG, so you can kill training halfway and
resume on the exact same curve.

## 📹 Metadata
- **YouTube title:** `MNIST on the M4 GPU: 97.7% in 2 seconds (and when the CPU wins) | PyTorch from Zero #12`
- **Thumbnail (3–5 words):** `CPU vs M4 GPU`
- **Short (which beat, 45–60 s):** The sync trap. "4 ms. No wait, 817 ms." One line of code makes your GPU benchmark honest.
- **X post:** "Timed 20 matmuls on my M4's GPU: 4.4 ms. Added torch.mps.synchronize(): 817 ms. GPU calls return when work is queued, not done. Every GPU benchmark you've seen without a sync is a lie. (Also: MNIST MLP, 97.7%, 0.34 s/epoch on MPS, 0.52 on CPU.)"
- **Sources:** PyTorch: Learn the Basics (https://docs.pytorch.org/tutorials/beginner/basics/intro.html); MNIST idx format (LeCun et al.); MPS backend docs
