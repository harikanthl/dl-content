# EP009 · M01E07 · Linear regression with raw autograd

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 7-9 min (part 7 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e07_linear_regression_with_raw_autograd.py` · **Explainer:** `explainers/scenes/m01e07_linear_regression_with_raw_autog.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "This for loop is how every LLM gets trained. Not a simplified version. The same four lines: predict,
> measure the error, get gradients, step downhill. Today there's no `nn`, no `optim`, nothing hiding
> it. Two numbers, and we watch them find the answer."

## ❓ The question (why this matters, 45 s)
`nn.Module`, `torch.optim.AdamW`, `loss.backward()`, `optimizer.step()`: by M06 we'll be training a
GPT with all of it. Every one of those is a convenience wrapper around this loop. If you've written
it by hand once, you know what `zero_grad` protects you from, why the update sits under `no_grad`, and
what a learning rate that's too big looks like. Those are the questions that come up when a real run
diverges at 3am.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: walking down a foggy hill.** You can't see the bottom. You can feel the slope under your
> feet. Take a step downhill, proportional to how steep it is. Repeat. The gradient is the slope.
> The learning rate is your stride. Too short and you're out there all night. Too long and you leap
> over the valley and land higher on the other side, and the next leap is even bigger.

**The one picture:** a scatter of 200 noisy points around `y = 3x + 2`, and a line that starts flat
at `w = 0, b = 0` and swings into place over ~20 steps. Next to it, the loss curve on a log scale
dropping and flattening at the noise floor.

## 📐 The math (2 min)
```
model:     ŷ_i = w·x_i + b
loss:      L = (1/n) Σ (ŷ_i - y_i)²                          (MSE)
gradients: ∂L/∂w = (2/n) Σ (ŷ_i - y_i)·x_i
           ∂L/∂b = (2/n) Σ (ŷ_i - y_i)
update:    w ← w - lr·∂L/∂w,   b ← b - lr·∂L/∂b
```
We don't type the gradient formulas. `loss.backward()` computes them (M01E06). They're on screen so
you can check the first row of the dry run by eye: at `w = b = 0`, `ŷ = 0`, so `∂L/∂b = -2·mean(y) ≈ -3.29`.

**Why it converges to the right answer (the correctness argument):** MSE for a linear model is a
convex bowl in `(w, b)`. It has one minimum, and it's the least-squares solution, which you can also
get in closed form with `torch.linalg.lstsq`. So we have an independent answer to check against. If
gradient descent lands on the lstsq numbers, the loop is right.

### Dry run: first steps, lr = 0.1 (measured on M4 Mac mini)
| step | w | b | loss | ∂L/∂w | ∂L/∂b |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 15.1670 | -7.7175 | -3.2857 |
| 1 | 0.7717 | 0.3286 | 8.9867 | -5.7117 | -2.7906 |
| 2 | 1.3429 | 0.6076 | 5.4286 | -4.2347 | -2.3524 |
| 3 | 1.7664 | 0.8429 | 3.3574 | -3.1456 | -1.9708 |

Check one by eye: `w_1 = 0 - 0.1 × (-7.7175) = 0.7717`. Gradients shrink as we approach the bottom,
so steps shrink on their own.

## 💻 Code it (from scratch, 4 min)
```python
def make_data(n=200, w_true=3.0, b_true=2.0, noise=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g) * 4 - 2                    # x in [-2, 2]
    y = w_true * x + b_true + noise * torch.randn(n, generator=g)
    return x.to(device), y.to(device)

def train(x, y, lr=0.1, steps=200):
    w = torch.zeros((), device=device, requires_grad=True)
    b = torch.zeros((), device=device, requires_grad=True)
    losses = []
    for _ in range(steps):
        pred = w * x + b                    # 1. forward
        loss = ((pred - y) ** 2).mean()     # 2. loss
        loss.backward()                     # 3. gradients into w.grad, b.grad
        with torch.no_grad():               # 4. step: not part of the model
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None                       # 5. empty the tip jar
        b.grad = None
        losses.append(loss.item())          # .item(), not loss: don't keep the graph
    return w.detach(), b.detach(), losses
```
Then `closed_form` with `torch.linalg.lstsq` and a matplotlib plot saved to `code/M01-pytorch/out/e07_loss.png`.

## ✅ The receipt (1 min)
```python
torch.testing.assert_close(w_gd, w_lstsq, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(b_gd, b_lstsq, rtol=1e-4, atol=1e-4)
```
Gradient descent and the closed-form least-squares solution agree to 4 decimals. Plus: recovered
`w, b` within 0.1 of `3, 2`, and the same loop without zeroing grads does **not** converge.
`pytest ... -q` → **3 passed** (measured on M4 Mac mini, MPS).

## 📊 Measure (1 min)
Measured on M4 Mac mini, actual script output:
```
step   0  loss 15.1670
step   5  loss 1.4129
step  10  loss 0.3862
step  25  loss 0.2789
step 199  loss 0.2787
gradient descent: w=3.0237 b=1.9603
closed form     : w=3.0237 b=1.9603
final loss 0.2787  vs  loss of true w=3,b=2: 0.2813
```
Two talking points:
- **We didn't get exactly 3 and 2, and that's correct.** The data is noisy. 3.0237 and 1.9603 are the
  best line *for these 200 points*. Our loss (0.2787) is even *lower* than the true line's (0.2813).
  The model fit a bit of the noise. That's overfitting in miniature.
- **Learning-rate sweep** (same data, 200 steps): lr 0.01 → w=3.0005, b=1.911 (still walking);
  lr 0.1 and 0.5 → converged; lr 0.8 → w = -1.1e16 (diverged); lr 1.0 → NaN.

Show `out/e07_loss.png`: log-scale loss, drops for ~20 steps, flat after.

## ⚠️ Gotchas
- **Forgetting to zero grads.** Measured: without `w.grad = None`, w goes 0.77 → 5.75 → 0.75 → 3.59 →
  2.23 over 50 steps. It bounces instead of settling, because each step uses the sum of every past
  gradient.
- **Updating outside `no_grad`:** `w -= lr * w.grad` raises `RuntimeError: a leaf Variable that requires
  grad is being used in an in-place operation.` Autograd refuses to record the update as part of the model.
- **`w = w - lr * w.grad` (not in-place) is worse:** no error, but `w` becomes a new non-leaf tensor.
  Next step `w.grad` is None and you crash with a confusing `TypeError`.
- **`losses.append(loss)`** keeps 200 graphs alive. `loss.item()` is a Python float.
- **lr too big → NaN.** The loss climbs before it NaNs. If your loss goes *up* in the first few steps,
  lower the lr before you debug anything else.
- **`torch.linalg.lstsq` on MPS** isn't the point here; we run it on CPU. Small solvers are often CPU-only
  or slower on GPU.

## 🎤 Interview lines
- *"A training step is forward, loss, backward, update, zero grads. Optimizers and modules just
  package that loop."*
- *"The update goes under `no_grad` because it isn't part of the function we're differentiating; we're
  changing the parameters, not computing with them."*
- *"For linear regression with MSE the loss is convex, so gradient descent should converge to the
  least-squares solution. I check it against `lstsq`."*
- *"If loss increases early, the learning rate is too high. Divergence shows up as growing loss, then inf,
  then NaN."*

## 🔗 Next
**Bridge (say it):** "Two parameters, managed by hand. GPT-2 has 124 million. We are not zeroing those
one by one. Next: `nn.Module` and `nn.Parameter`, the bookkeeping that tracks every weight for you."
→ **M01E08: nn.Module and nn.Parameter.**

## 📹 Metadata
- **YouTube title:** `Gradient descent from scratch in PyTorch: linear regression with raw autograd`
- **Thumbnail (3-5 words):** `NO nn. NO optim.`
- **Short (which beat, 45-60 s):** the learning-rate sweep. lr 0.1 converges, 0.8 explodes to -1e16, 1.0 is NaN. Caption: "One number decides if it trains."
- **X post:** Linear regression with no nn and no optim, just two tensors and loss.backward(). Gradient descent lands on w=3.0237, b=1.9603, the exact least-squares answer. Forget to zero the grads and w bounces 0.77 → 5.75 → 0.75 → 3.59 instead. Code + video:
- **Sources:** [PyTorch: Learn the Basics, optimization loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) · [torch.linalg.lstsq](https://docs.pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) · [torch.no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html)
