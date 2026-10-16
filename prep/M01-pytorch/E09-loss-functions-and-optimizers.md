# EP011 · M01E09 · Loss functions and optimizers

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e09_loss_functions_and_optimizers.py` · **Explainer:** `explainers/scenes/m01e09_loss_functions_and_optimizers.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "torch.optim.SGD is about ten lines of math. I'll write it by hand, run it next to
> PyTorch's for 50 steps, and the losses won't be close. They'll be identical. Bit for bit."

## ❓ The question (why this matters, 45 s)
Every LLM is trained by the same two objects: a loss (cross-entropy over the next token)
and an optimizer (AdamW, in GPT-2, Llama 3, DeepSeek-V3, all of them). If you think of
them as black boxes, you can't debug a loss spike, you can't resume a run correctly (E13),
and you can't read the AdamW episode in M06. Today we open both boxes. Cross-entropy is
two lines. SGD with momentum is five.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5, the loss:** a teacher's red pen. One number that says how wrong you were.
> MSE is "how far off was your guess", for numbers. Cross-entropy is "how surprised were
> you by the right answer", for choices. If you said 99% cat and it was cat, tiny
> surprise. If you said 1% cat and it was cat, huge surprise: `-log(0.01) = 4.6`.
>
> **ELI5, momentum:** a ball rolling down a hill. Plain SGD is a hiker who only looks at
> the slope under their feet each step. Momentum is a ball: it remembers which way it was
> already going. In a long narrow valley, the hiker zig-zags wall to wall. The ball
> builds up speed along the valley floor and the zig-zags cancel out.

The picture: a narrow elliptical bowl. SGD path zig-zags. Momentum path curves smoothly
down the middle. That's the explainer clip.

## 🪤 The trap
**The order of the five lines.** `zero_grad → forward → loss → backward → step`.
PyTorch *adds* new gradients into `p.grad`. It never overwrites. Skip `zero_grad` and
step 2 uses grad₁ + grad₂, step 10 uses the sum of ten. The receipt
`test_grads_accumulate_without_zero_grad` shows it: two `backward()` calls, grad is exactly
2x. (This is a feature, not a bug: it's how gradient accumulation works in M06. But only
when you mean it.)

**Second trap, cross-entropy takes logits.** `F.cross_entropy` does the softmax inside.
Softmax first and you softmax twice. It still trains, badly. On camera:
`cross_entropy([2, 1, 0.1], 0)` = 0.417, `cross_entropy(softmax([2, 1, 0.1]), 0)` = 0.802.
We'll prove in E14 that this bug puts a hard floor under your loss.

## 📐 The math (derive on screen, 2–3 min)
**MSE.** `L = mean((ŷ − y)²)`. Shapes: `ŷ (B,)`, `y (B,)`. Same shape, or broadcasting
eats you (E14, bug 1).

**Cross-entropy.** logits `z (B, C)`, target `t (B,)` of class ids.
1. `p = softmax(z)`, so `p_t = exp(z_t) / Σ_j exp(z_j)`.
2. `L = −log p_t` (surprise at the right answer).
3. Expand the log: `L = −z_t + log Σ_j exp(z_j) = logsumexp(z) − z_t`.
   Why bother: we never form `p`. `logsumexp` subtracts `max(z)` first, so `z = 1000`
   doesn't overflow. Naive `log(softmax([1000, 0]))` gives `[0, -inf]`. Ours stays finite.
4. Gradient (say it, it's beautiful): `∂L/∂z = p − onehot(t)`. Predicted minus truth.

**SGD with momentum, PyTorch's version.** (lr η, momentum μ, weight decay λ)
1. `g = ∇L + λ·θ`           weight decay folded into the gradient (this is L2, not AdamW's decoupled decay)
2. `b = g` on the first step, then `b = μ·b + g`
3. `θ = θ − η·b`

Note: CS231n and Sutskever et al. write `v = μv − η·g; θ = θ + v`. With a constant lr it's
the same trajectory (v = −η·b). PyTorch's form keeps the lr outside the buffer, so changing
the lr mid-run (schedulers, M06) doesn't get "remembered" in the velocity. Mention this, it
comes up in interviews and in the docs' own note.

Unrolled: `b_t = g_t + μ g_{t−1} + μ² g_{t−2} + ...`. With μ = 0.9, the effective step is
about `1/(1−μ) = 10x` the gradient along a consistent direction. Zig-zag components
alternate sign and cancel.

## 💻 Code it (from scratch, 4–5 min)
```python
def cross_entropy(logits, target):            # (B, C), (B,) int64
    lse = torch.logsumexp(logits, dim=-1)                         # (B,)
    correct = logits.gather(1, target[:, None]).squeeze(1)        # (B,) logit of right class
    return (lse - correct).mean()

class MySGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = list(params)            # list(): model.parameters() is a generator
        self.lr, self.mu, self.wd = lr, momentum, weight_decay
        self.buf = [None] * len(self.params)  # one velocity per parameter

    @torch.no_grad()                          # the update is not part of the graph
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.add(p, alpha=self.wd) if self.wd else p.grad
            if self.mu:
                if self.buf[i] is None:
                    self.buf[i] = g.clone()   # first step: b = g
                else:
                    self.buf[i].mul_(self.mu).add_(g)
                g = self.buf[i]
            p.add_(g, alpha=-self.lr)         # in place: the optimizer and module share this object

    def zero_grad(self):
        for p in self.params:
            p.grad = None                     # PyTorch default since 2.0
```
Then `run("mine")` and `run("torch")`: same seed, same data, lr 0.1, μ 0.9, λ 1e-4,
the five-line loop.

## ✅ The receipt (1 min)
```python
mine, m1 = run("mine", steps=50)
ref,  m2 = run("torch", steps=50)
assert mine == ref                         # Python list equality on floats. Exact.
for p1, p2 in zip(m1.parameters(), m2.parameters()):
    assert torch.equal(p1, p2)
```
Plus `cross_entropy` matches `F.cross_entropy` (including logits scaled by 5), `mse`
matches `F.mse_loss`, and the grad-accumulation test. `pytest -q` → **4 passed**.

## 📊 Measure (1 min)
Measured on M4 Mac mini, CPU:
- Loss over 50 steps: 1.110 → 0.261, identical in both optimizers at every step.
- **The war story for camera:** my first version wrote `p.grad + wd * p` and
  `p.sub_(lr * g)`. Mathematically identical. It matched to a max difference of
  **1.2e-7** in the loss, not 0. Floating-point addition isn't associative, and
  `p.add_(g, alpha=-lr)` rounds once where `p.sub_(lr * g)` rounds twice. Switching to the
  `alpha=` form, which is what PyTorch does internally, made it bit-exact. That's the
  difference between "close" and "the same", and it's worth 30 seconds.

## ⚠️ Gotchas
- **Missing `zero_grad`.** No error. Effective lr grows every step. (E14 shows the
  grad norm climbing from 0.46 to 6.4 in 20 steps.)
- **Softmax before `CrossEntropyLoss`.** No error. Trains slow, plateaus high.
- **Target dtype.** `cross_entropy` wants class ids as `int64`. Pass floats with a
  `(B,)` shape and you get `RuntimeError: expected target dtype to be Long or Byte, but got Float`.
  Pass a `(B, C)` float target and PyTorch treats it as class *probabilities*. Different
  loss, no error.
- **`params` as a generator.** If you store `model.parameters()` without `list()`, the
  first loop exhausts it and every later `step()` does nothing.
- **Out-of-place update.** `p = p - lr * g` rebinds a local variable. The model never
  changes. Must be in place: `p.add_(...)` or `p.data -= ...`.
- **Update outside `no_grad`.** `RuntimeError: a leaf Variable that requires grad is being
  used in an in-place operation.`

## 🎤 Interview lines
- *"Cross-entropy is logsumexp of the logits minus the correct logit. Computing it that
  way avoids forming softmax probabilities, so it's numerically stable. The gradient with
  respect to the logits is softmax minus one-hot."*
- *"PyTorch accumulates gradients into .grad, so you zero them each step. That's also
  why gradient accumulation across micro-batches is free: just don't zero."*
- *"PyTorch's SGD momentum keeps the lr outside the velocity buffer: b = μb + g, θ -= ηb.
  The Sutskever form puts lr inside. Same thing at constant lr, different under a schedule."*
- *"SGD's weight_decay is L2 added to the gradient. AdamW decouples it and applies it
  directly to the weights. With SGD they're equivalent up to an lr factor, with Adam they're not."*

## 🔗 Next
We have a loss and an optimizer, and a five-line loop. Next, M01E10: we wrap that loop
into a reusable `train()` with a proper validation pass, and find out why `model.eval()` and
`torch.no_grad()` are two different things.

## 📹 Metadata
- **YouTube title:** `I wrote torch.optim.SGD by hand. It matches bit for bit. | PyTorch from Zero #9`
- **Thumbnail (3–5 words):** `SGD BY HAND = EXACT`
- **Short (which beat, 45–60 s):** The war story. "Mathematically identical, off by 1e-7. One keyword fixed it."
- **X post:** "Hand-wrote SGD with momentum. First version matched torch.optim.SGD to 1e-7. Not 0. Because p.sub_(lr*g) rounds twice and p.add_(g, alpha=-lr) rounds once. Fixed it, now bit-identical for 50 steps. Floating point is not algebra."
- **Sources:** PyTorch documentation, `torch.optim.SGD` (algorithm box) and `torch.nn.functional.cross_entropy` (https://docs.pytorch.org/docs/stable/index.html)
