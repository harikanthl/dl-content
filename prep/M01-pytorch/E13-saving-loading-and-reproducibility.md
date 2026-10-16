# EP015 · M01E13 · Saving, loading and reproducibility

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e13_saving_loading_and_reproducibility.py` · **Explainer:** `explainers/scenes/m01e13_saving_loading_and_reproducibili.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "Most people's checkpoints are wrong. They save the model, resume, and the loss curve
> quietly changes. I'm going to kill a training run halfway, resume it, and get the same
> 64 loss values. Not close. Equal."

## ❓ The question (why this matters, 45 s)
Long runs die. Spot instances get preempted, a node fails, you hit Ctrl-C by accident.
Meta's Llama 3 paper reports hundreds of interruptions over its 54-day pretraining run.
Checkpoint-and-resume isn't an edge case, it's the normal path.

And if your resume is subtly wrong, you can't tell. The loss still goes down. You just
trained a different model than you think, and every comparison you make afterwards
("did my change help?") is noise. In M06 we train nanoGPT on RunPod by the hour. This
episode is what makes that safe.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: saving a video game.**
> A bad save stores your character's level. A real save stores your level, your inventory,
> where you're standing, and the random seed for what's behind the next door.
>
> A training run is the same. Four things change while you train, so four things go in
> the save file:
> 1. **The weights** (your level): `model.state_dict()`
> 2. **The optimizer state** (your inventory): Adam's running averages `m` and `v`, and
>    the step count. This is momentum. Lose it and the optimizer "forgets which way it was
>    going".
> 3. **Where you are**: the epoch (and in M06, the step, for the lr schedule).
> 4. **The dice**: the global RNG (dropout masks) and the DataLoader's generator
>    (shuffle order).

The picture: a timeline of 128 loss values. A red "X" at step 64. Three restarts drawn on
top: full checkpoint lands exactly on the original line; no-optimizer bumps off it;
no-RNG starts on a different value immediately.

## 🪤 The trap
**Saving only the model.** It's what `torch.save(model.state_dict(), "model.pt")`
tutorials teach, and it's correct for *inference*. For *resuming*, it's wrong in a way you
can't see. Measured on camera (128 steps, crash after step 64):

| resume with | first resumed loss (target 0.06421) | max diff vs uninterrupted |
|---|---|---|
| everything | 0.06421 | **0** |
| no optimizer state | 0.06421 | 2.8e-2 |
| no RNG state | 0.08918 | 9.5e-2 |

Read the middle row carefully: the first loss is *right*. The forward pass doesn't use the
optimizer. The damage starts one step later, when a fresh Adam with empty `m` and `v`
takes a full-size step. That's what makes this bug sneaky: the spot check passes.

## 📐 The math (derive on screen, 2–3 min)
**Why Adam's state matters.** Adam keeps, per parameter:
`m_t = β₁ m + (1−β₁) g`, `v_t = β₂ v + (1−β₂) g²`, and bias-corrects with the step `t`:
`m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`, update `−lr · m̂/(√v̂ + ε)`.
Reset `m = v = 0, t = 0` and the bias correction says "this is step 1". Step 1 of Adam is
roughly `−lr · sign(g)`: every parameter moves by a full `lr`, regardless of how noisy its
gradient was. Mid-training, that's a shove.

**How big is a checkpoint.** Our model: `20·64+64 + 64·2+2 = 1,474` params.
- weights: 1,474 × 4 B ≈ 5.9 KB
- Adam `m` + `v`: 2 × 5.9 KB ≈ 11.8 KB
- two RNG states (CPU generator + loader generator): ~5 KB each
- Total predicted ≈ 33 KB. **Measured: 32.5 KB.**

Scale it: Adam's state is 2x the parameter count. For an 8B model with fp32 `m` and `v`,
that's 64 GB of optimizer state, more than the bf16 weights (16 GB). Checkpoints of real
LLM runs are mostly optimizer.

**Why resume at an epoch boundary.** The DataLoader draws its shuffle permutation at the
start of each epoch from its generator. Saving the generator at the boundary means the
resumed epoch draws the same permutation. Resuming mid-epoch needs the step index too
(skip that many batches); M06 does it with a token offset instead.

## 💻 Code it (from scratch, 4–5 min)
```python
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)  # manual_seed covers MPS/CUDA too

def save_checkpoint(path, model, opt, epoch, loader_gen):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,                          # the NEXT epoch to run
        "torch_rng": torch.get_rng_state(),      # dropout's dice
        "loader_rng": loader_gen.get_state(),    # shuffle's dice
    }, path)

# resume: build everything fresh, exactly as a new process would, then overwrite
state = torch.load(ckpt)                         # weights_only=True by default (torch >= 2.6)
model.load_state_dict(state["model"])
opt.load_state_dict(state["optimizer"])          # AFTER the optimizer is built on THIS model's params
torch.set_rng_state(state["torch_rng"])
gen.set_state(state["loader_rng"])
start = state["epoch"]
```
`run(epochs, save_to=..., stop_after=2)` simulates the crash. `run(epochs, resume_from=...)`
is the restart. Compare the concatenated loss lists with `==`.

Talk through `weights_only=True`: `torch.load` used to unpickle arbitrary Python objects,
which means a downloaded checkpoint could run code on your machine. Since 2.6 the default
only allows tensors and plain containers. Our checkpoint is all tensors, dicts and ints,
so it loads under the safe default. Keep your checkpoints that way.

## ✅ The receipt (1 min)
```python
full   = run(4)                                  # uninterrupted: 128 losses
first  = run(4, save_to=ck, stop_after=2)        # dies after epoch 2: 64 losses
second = run(4, resume_from=ck)                  # picks up: 64 losses
assert first + second == full                    # exact float equality
```
Plus: without optimizer state the curves differ, without RNG state they differ, and the
checkpoint loads with `weights_only=True` and has exactly the 5 keys. `pytest -q` →
**4 passed** in about 1 s.

## 📊 Measure (1 min)
Measured on M4 Mac mini:
- Resumed vs uninterrupted, 64 steps after the crash: **max diff 0.0** (CPU).
- Checkpoint size: **32.5 KB** for 1,474 params, matching the arithmetic above.
- Same seed, trained twice, 50 Adam steps on a 256→512→10 MLP: max weight difference
  **0.0 on CPU, 0.0 on MPS**.

Be honest about that last one: it's 0.0 *for this model on this machine*. That's not a
guarantee. GPU kernels that use atomic adds (scatter, index_add, some backward passes)
can sum in a different order each run. On CUDA, `torch.use_deterministic_algorithms(True)`
(plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`) makes PyTorch error on nondeterministic ops or swap
in deterministic ones. Different GPU, different PyTorch version, different batch size:
all can change the bits. "Reproducible" means same code, same hardware, same versions.

## ⚠️ Gotchas
- **Saving only `model.state_dict()` for a resume.** Silent. See the table.
- **Saving the whole model object** (`torch.save(model)`). Pickles your class by import
  path; rename a file and it won't load. Also fails `weights_only=True`. Save state_dicts.
- **Loading optimizer state before building the optimizer on the right params.** The
  optimizer state is keyed by parameter *position*. Change the model's layer order and it
  loads without error into the wrong tensors.
- **Device of the loaded checkpoint.** A checkpoint saved on MPS or CUDA loads tensors
  back onto that device. Use `torch.load(path, map_location="cpu")` on a different machine.
- **Checkpoint saved from `torch.compile` or DDP.** Keys get `_orig_mod.` or `module.`
  prefixes. `load_state_dict` fails with `Missing key(s)` / `Unexpected key(s)`. Strip the
  prefix. (M06 and M10.)
- **Shuffling from the global RNG.** If the DataLoader has no generator of its own, data
  order depends on how many random numbers dropout used. Give it its own.
- **Overwriting the only checkpoint.** If the process dies *during* `torch.save`, you have
  a corrupt file and no backup. Write to `ckpt.tmp`, then `os.replace` it. Cheap insurance.

## 🎤 Interview lines
- *"A resumable checkpoint has model state, optimizer state, the step or epoch, and RNG
  states, both the global one and the data loader's generator. Missing optimizer state is
  the common bug: the loss doesn't jump at the first step, so nobody notices."*
- *"For Adam, the optimizer state is twice the parameter count, so for large models the
  checkpoint is mostly optimizer."*
- *"Bitwise reproducibility needs same seeds, same hardware, same library versions, and
  deterministic kernels. On GPU, atomics in some kernels break it unless you turn on
  deterministic algorithms."*
- *"I save state_dicts, not pickled modules, and load with weights_only=True, because
  unpickling an untrusted checkpoint can execute code."*

## 🔗 Next
We can now train, evaluate, save and resume. What we can't do yet is figure out why a
model *isn't* training. Next, M01E14: four broken models, four debugging tools, on camera.

## 📹 Metadata
- **YouTube title:** `Your PyTorch checkpoints are probably wrong (here's the fix) | PyTorch from Zero #13`
- **Thumbnail (3–5 words):** `RESUME ≠ RESTART`
- **Short (which beat, 45–60 s):** The trap table. Kill at step 64, resume three ways, only one lands on the original curve.
- **X post:** "Killed a training run at step 64. Resumed 3 ways. Full checkpoint: identical curve, diff 0.0. Without optimizer state: first loss correct (!), then drifts. Without RNG state: wrong from step 1. Save the optimizer and both RNGs. 32 KB of insurance."
- **Sources:** PyTorch documentation, "Saving and Loading Models" and "Reproducibility" notes (https://docs.pytorch.org/docs/stable/index.html); Llama 3 paper (Dubey et al., 2024) on training interruptions
