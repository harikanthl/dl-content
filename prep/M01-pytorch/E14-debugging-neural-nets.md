# EP016 · M01E14 · Debugging neural nets

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 12–15 min
**Code:** `code/M01-pytorch/e14_debugging_neural_nets.py` · **Explainer:** `explainers/scenes/m01e14_debugging_neural_nets.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "Neural nets don't crash when they're wrong. They just train worse. I've broken this
> model four different ways, none of them throw an error, and one of them makes the loss
> look *better*. Let's catch all four."

## ❓ The question (why this matters, 45 s)
Karpathy's line: neural net training "fails silently". A normal program with a bug
crashes. A model with a bug still produces a loss, the loss still goes down, and you ship
something 5% worse without knowing. Every one of the 200 episodes after this has a moment
where something doesn't work. This episode is the toolkit we reach for every time:
shape asserts, overfit one batch, gradient norms, forward hooks. Four tools, four bugs,
each bug picked because it's one I (and you) will actually write.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a car that drives but drives wrong.**
> If the engine doesn't start, easy, you know where to look. That's a crash.
> The hard ones: the car drives, just pulls a bit left, or the speedometer reads 10% high.
> A mechanic doesn't guess. They have specific tests: check tire pressure (shape asserts),
> put it on rollers and see if it can hit top speed at all (overfit one batch), watch the
> temperature gauge (gradient norms), plug in the diagnostic port that tells you which
> sensor failed first (forward hooks).

The picture: a 2x2 grid. Each cell: the bug, the symptom (what you'd see), the tool,
the tell. That grid is also the end card.

## 🪤 The trap (four of them, that's the episode)
| # | bug | what you see | tool | the tell |
|---|---|---|---|---|
| 1 | pred `(B,1)` vs target `(B,)` | loss goes down, model is useless | shape assert | `pred (256, 1) vs target (256,)` |
| 2 | softmax before `cross_entropy` | trains, plateaus high | overfit one batch | loss stuck at exactly 1.4612 |
| 3 | no `zero_grad` | loss looks *better* early | grad norm per step | norm climbs 0.46 → 6.38 |
| 4 | `log(softmax(x))` | NaN / -inf from nowhere | forward hooks + `isfinite` | names module `"2"` |

## 📐 The math (derive on screen, 2–3 min)
**Bug 1, why it learns the mean.** `pred (B,1) − y (B,)` broadcasts to `(B, B)` with
entry `pred_i − y_j`. The loss is `mean_{i,j} (pred_i − y_j)²`. For each `i`, that's a
quadratic in `pred_i`, minimized at `pred_i = mean_j(y_j)`. So the optimum is: predict the
mean of the targets for every input, ignoring the input. The true MSE then equals `var(y)`.
Measured: true MSE **23.868**, `var(y)` **23.868**. Same number to three decimals. The
model learned exactly what the bug told it to.

**Bug 2, why there's a floor.** `cross_entropy` applies softmax to its input. If the input
is already probabilities `p ∈ [0,1]` summing to 1, the best you can feed it is one-hot:
`p_t = 1`, rest 0. Then the loss is
`−log(e¹ / (e¹ + 9·e⁰)) = log(1 + 9/e) = 1.4612` for 10 classes.
No amount of training goes below that. Measured after 500 Adam steps on one batch:
**1.4612**. The correct loss on the same batch: **0.0001**.
General formula for C classes: `log(1 + (C−1)/e)`. For a 50k-token LLM vocabulary that's
`log(1 + 49999/e) ≈ 9.8`, which looks like a model that's "just learning slowly".

**Bug 3, why grads explode.** Without zeroing, `p.grad` at step k is `Σ_{i≤k} g_i`.
If gradients point roughly the same way, the norm grows like `k·|g|`. SGD's effective lr
grows linearly with the step count. Early on, that looks like faster learning (loss 1.383
vs 2.194 at step 20). Later, it diverges.

**Bug 4, why NaN.** `softmax([1000, 0]) = [1, e⁻¹⁰⁰⁰]`, and `e⁻¹⁰⁰⁰` underflows to `0`
in float32. `log(0) = −inf`. Then `−inf · 0` or `inf − inf` downstream gives `NaN`. The fix,
`F.log_softmax`, computes `x − logsumexp(x)` and never forms the tiny probability.

## 💻 Code it (from scratch, 4–5 min)
Tool 1, shape assert:
```python
assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
```
Tool 2, overfit one batch:
```python
def overfit_one_batch(model, x, y, loss_fn, steps=500, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()                         # healthy: ~0. broken: stuck.
```
Tool 3, gradient norm:
```python
def grad_norm(model):
    return torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                          if p.grad is not None)).item()
```
(`torch.nn.utils.clip_grad_norm_` returns this same number. In M06 we log it every step.)

Tool 4, NaN-finding hooks:
```python
def add_nan_hooks(model):
    bad, handles = [], []
    for name, m in model.named_modules():
        def hook(mod, inp, out, name=name):     # name=name freezes the loop variable
            if not torch.isfinite(out).all() and not bad:
                bad.append(name)                # first module to go non-finite
        handles.append(m.register_forward_hook(hook))
    return bad, handles
```
Hooks fire innermost-first in forward order, so the first name recorded is the culprit,
not the parent `Sequential` (which also outputs NaN, but fires after its children).

Demo order on camera: run each bug, show the symptom with no tool, then add the tool and
watch it point at the bug. Then the fix: `.squeeze(1)`, delete `.softmax(-1)`, add
`opt.zero_grad()`, use `F.log_softmax`.

## ✅ The receipt (1 min)
Five tests, one per bug plus the bug 1 fix:
- `test_bug1_...`: without the assert, true MSE > 1.0 after 300 steps; with it, the
  `AssertionError` names both shapes. `test_bug1_fixed`: with `.squeeze(1)`, loss < 0.1.
- `test_bug2_double_softmax_cannot_overfit`: correct CE < 0.01, double softmax ≥ 1.4612.
- `test_bug3_grad_norm_explodes_without_zero_grad`: final norm > 5x the healthy run.
- `test_bug4_hook_finds_the_naive_log_softmax`: culprit `"2"`, and `F.log_softmax` on
  `[1000, 0]` is finite.

`pytest -q` → **5 passed** in about 1.3 s.

## 📊 Measure (1 min)
Measured on M4 Mac mini, CPU:
- Bug 1: true MSE 23.868 = var(y) 23.868 (it predicts the mean). Fixed: < 0.1.
- Bug 2: overfit loss 0.0001 (correct) vs 1.4612 (double softmax, = the floor to 4 decimals).
- Bug 3: grad norm at steps 1/10/20: **0.46 / 0.42 / 0.39** healthy vs **0.46 / 3.93 / 6.38** broken.
  Loss at step 20: 2.194 healthy vs **1.383 broken**. The bug looks better. That's the
  lesson: a loss curve alone can lie. Log the grad norm.
- Bug 4: first non-finite module `"2"` (`NaiveLogSoftmax`), output row `[0, -inf, -inf, -inf]`.

## ⚠️ Gotchas
- `F.mse_loss` warns about bug 1 (`UserWarning: Using a target size (torch.Size([256]))
  that is different to the input size (torch.Size([256, 1]))`). A hand-written loss
  doesn't. And warnings scroll by. Asserts don't.
- **Late-binding closures.** Without `name=name` in the hook signature, every hook reports
  the *last* module's name. Python closures capture variables, not values.
- **Not removing hooks.** They fire forever, including during eval, and slow everything.
  Keep the handles, call `.remove()`.
- **`torch.autograd.detect_anomaly()`** finds NaNs in the *backward* pass and names the
  op. Very slow. Turn it on to hunt, off to train.
- **Overfit with dropout on.** Dropout fights memorization. If a model with dropout can't
  quite hit zero on one batch, try `model.eval()`-style dropout off before concluding it's broken.
- **Overfit one batch passing doesn't mean the model is right.** It means the plumbing
  works. It's the first test, not the last. (Karpathy's recipe: this, then a baseline,
  then scale.)

## 🎤 Interview lines
- *"Before training on the full data, I overfit a single batch. If the model can't drive
  loss to near zero on 32 examples, there's a bug in the model or the loss, and more data
  won't fix it."*
- *"Softmax before cross-entropy caps how low the loss can go: log(1 + (C−1)/e), about 1.46
  for 10 classes. If your loss plateaus at a suspiciously round-looking number, check for
  a double softmax."*
- *"I log the global gradient norm every step. Forgetting zero_grad, an lr that's too
  high, and loss spikes all show up there before they show up in the loss."*
- *"For NaNs I register forward hooks that check isfinite and record the first module that
  fails. For NaNs in backward, detect_anomaly."*
- *"Broadcasting a (B,1) prediction against a (B,) target makes a (B,B) loss whose
  minimizer is the target mean, so the model learns to ignore its input. Shape asserts
  at the loss boundary catch it."*

## 🔗 Next
That closes M01. We can build, train, save, resume and debug any PyTorch model. We've
been using dot products, gradients and softmax on faith. Next, M02E01: the math you actually
need, starting with vectors and dot products, the operation inside every layer we just debugged.

---

## 🎞️ Stitching M01 into "PyTorch in 1 Hour"
One long video cut from the 14 episodes. Target 60 min, about 4–5 min per chapter.
Per episode: keep the code-it beat and the receipt, keep the trap. Drop the face-cam hook
and "why this matters" (one shared intro covers it), and drop the metadata/outro.
Use each episode's bridge line as the spoken transition between chapters. Open with
E12's MNIST result as a cold open ("this is where we'll be in 50 minutes").

**Chapters (YouTube description):**
```
00:00 Cold open: 97.7% on MNIST, what we're building
01:00 1. Tensors: what they actually are
05:00 2. Creating tensors and dtypes
09:00 3. Indexing, views and strides
13:00 4. Broadcasting: the bug factory
17:30 5. Matmul, reductions and einsum
22:00 6. Autograd: how PyTorch computes gradients
27:00 7. Linear regression with raw autograd
31:00 8. nn.Module and nn.Parameter
35:00 9. Loss functions and optimizers
39:30 10. The training loop, done right
43:30 11. Dataset and DataLoader
47:30 12. MNIST MLP end to end on the M4 GPU
52:00 13. Saving, loading and reproducibility
55:30 14. Debugging neural nets
59:30 What's next: the math you actually need (M02)
```
Timestamps are planned, not measured. Re-time from the edit.

Throughline to say in the intro and pay off at the end: "Every chapter ends with a
receipt: a test that proves the from-scratch version matches PyTorch." Show the full
`pytest code/M01-pytorch -q` run (all files green) as the final shot.

## 📹 Metadata
- **YouTube title:** `4 bugs that don't crash your neural net (and how to catch each one) | PyTorch from Zero #14`
- **Thumbnail (3–5 words):** `LOSS STUCK AT 1.46?`
- **Short (which beat, 45–60 s):** Bug 2. Loss plateaus at 1.4612, derive `log(1 + 9/e)` on screen, delete one `.softmax()`, loss goes to 0.
- **X post:** "If your 10-class classifier's loss won't go below 1.46, you softmaxed before cross_entropy. The floor is log(1 + 9/e) = 1.4612. For a 50k vocab LLM it's ~9.8, which just looks like slow learning. Overfit one batch: healthy model hits 0.0001."
- **Sources:** Karpathy, Neural Networks: Zero to Hero (https://karpathy.ai/zero-to-hero.html) and "A Recipe for Training Neural Networks" (2019); PyTorch docs on hooks and `torch.autograd.detect_anomaly` (https://docs.pytorch.org/docs/stable/index.html)
