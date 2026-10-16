# EP008 · M01E06 · Autograd: how PyTorch computes gradients

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 8-9 min (part 6 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e06_autograd_how_pytorch_computes_gradients.py` · **Explainer:** `explainers/scenes/m01e06_autograd_how_pytorch_computes_gr.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "Call `.backward()` twice and your gradients double. Not a bug. PyTorch adds gradients, it never
> overwrites them. That one design choice is why every training loop has a `zero_grad()` in it, and
> today we'll see exactly where the numbers come from."

## ❓ The question (why this matters, 45 s)
Training GPT-2 means computing the gradient of one number, the loss, with respect to 124 million
parameters, every step. Nobody writes those derivatives. Autograd records what you did in the
forward pass and replays it backwards with the chain rule. If you don't know what it records,
you'll hit the classic errors: "backward through the graph a second time," gradients that are
`None`, gradients that silently accumulate, memory that grows every step.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: breadcrumbs.** Every operation you do on a tensor that "requires grad" drops a breadcrumb
> saying what it did and what it needs to remember. `.backward()` walks the breadcrumbs from the end
> back to the start, and at each one asks "if my output nudges by 1, how much do my inputs nudge?"
> Multiply those nudges along the path. That's the chain rule.

> **And `.grad` is a tip jar, not a scoreboard.** Every backward pass drops more in. Nobody empties
> it unless you do.

The vocabulary, one line each:
- **leaf**: a tensor you created (`requires_grad=True`). Only leaves keep `.grad`.
- **grad_fn**: the breadcrumb on a non-leaf. `out.grad_fn` → `PowBackward0`.
- **backward()**: walk the graph, add into each leaf's `.grad`, then **free the graph**.
- **no_grad()**: stop dropping breadcrumbs (eval, weight updates).
- **detach()**: same data, cut off from the graph.

## 📐 The math (2-3 min)
Function: `f(a, b) = (a·b + b)²`. Break it into nodes:
```
c = a·b        dc/da = b        dc/db = a
d = c + b      dd/dc = 1        dd/db = 1        (b feeds d twice: through c AND directly)
f = d²         df/dd = 2d
```
Chain rule:
```
df/da = df/dd · dd/dc · dc/da          = 2d · b
df/db = df/dd · (dd/dc · dc/db + dd/db) = 2d · (a + 1)     <- paths ADD
```
**That "paths add" line is the reason `.grad` accumulates.** When a value is used twice, its gradient
is the sum over both uses. Accumulating is the correct primitive; it just also means a second
`.backward()` adds to the first.

### Dry run: a = 3, b = 2
| node | forward value | backward: d f / d node | how |
|---|---|---|---|
| f = d² | 64 | 1 | start |
| d = c + b | 8 | 2·8 = **16** | df/dd = 2d |
| c = a·b | 6 | 16 · 1 = **16** | through `+` unchanged |
| a | 3 | 16 · b = **32** | through `·`, multiply by the other input |
| b | 2 | 16 · a + 16 = 48 + 16 = **64** | two paths, add them |

Autograd says `a.grad = 32, b.grad = 64`. Matches.

**Finite differences, the independent check:**
```
df/dx ≈ (f(x + h) - f(x - h)) / 2h       error ~ h², use float64 and h = 1e-6
```

## 💻 Code it (from scratch, 3 min)
```python
def f(a, b):
    c = a * b
    d = c + b
    return d ** 2

def grad_by_hand(a, b):
    d = a * b + b
    return 2 * d * b, 2 * d * (a + 1)

def grad_finite_diff(fn, a, b, h=1e-6):
    da = (fn(a + h, b) - fn(a - h, b)) / (2 * h)
    db = (fn(a, b + h) - fn(a, b - h)) / (2 * h)
    return da, db

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
out = f(a, b)
out.backward()
a.grad, b.grad          # 32., 64.
```
Then the three traps live: backward again (doubles), backward on the same `out` (error),
`.grad` on a non-leaf (None).

## ✅ The receipt (1 min)
```python
torch.testing.assert_close(torch.tensor(auto), torch.tensor(hand))              # autograd == calculus
torch.testing.assert_close(hand_f64, finite_diff_f64, rtol=1e-6, atol=1e-6)     # calculus == numerics
assert torch.autograd.gradcheck(f, (a, b))                                       # PyTorch's own checker
```
Three points tested, plus `gradcheck` on random float64 vectors, plus the accumulation test
(`32 + 32 = 64`), plus the same gradient on MPS. `pytest ... -q` → **4 passed** (measured on M4 Mac mini).

## 📊 Measure (1 min)
Measured on M4 Mac mini, actual script output:
```
f = 64.0
graph: <PowBackward0> -> ((<AddBackward0>, 0),)
autograd   : 32.0 64.0
by hand    : (32.0, 64.0)
finite diff: (32.00000000092018, 64.00000000184036)
after 2nd backward, a.grad = 64.0 (doubled, not reset)
2nd backward: Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed)
under no_grad, requires_grad = False
is_leaf a, c: True False
on mps : (32.0, 64.0)
```
Finite differences agree to about 1e-9 relative. That's the h² error term with h = 1e-6, in float64.

## ⚠️ Gotchas
- **Gradients accumulate.** Second backward: `a.grad` goes 32 → 64. Every loop needs
  `opt.zero_grad()` (or `p.grad = None`). Forget it and training behaves like it has a broken,
  undamped momentum (you'll see it oscillate in M01E07).
- **Backward twice:** `RuntimeError: Trying to backward through the graph a second time (or directly
  access saved tensors after they have already been freed). Specify retain_graph=True...` The graph is
  freed after backward to save memory. Re-run the forward instead of reaching for `retain_graph`.
- **`.grad` is None on non-leaves.** `c = a * b; c.grad` → None plus a warning. Use `c.retain_grad()`
  if you really need it.
- **`.backward()` on a non-scalar:** `RuntimeError: grad can be implicitly created only for scalar
  outputs`. Reduce to a loss first (or pass `gradient=`).
- **Finite differences in float32 are garbage.** Near x = 3, float32 numbers are 2.4e-7 apart, so
  h = 1e-6 is only about four representable steps and the difference is mostly rounding. Measured on
  M4 Mac mini: float32 central difference for df/da at (3, 2) gives **30.52**, not 32. Use float64,
  which means **CPU, because MPS has no float64**.
- **Memory leak pattern:** `losses.append(loss)` keeps every step's graph alive. Use `loss.item()`.
- **`detach()` shares storage.** Modify the detached tensor in place and the original changes too.

## 🎤 Interview lines
- *"Autograd records a graph of operations during the forward pass; backward traverses it in reverse
  topological order applying the chain rule, and accumulates into the leaves' `.grad`."*
- *"Gradients accumulate because a tensor used in several places needs the sum of gradients from each
  path. That's also what makes gradient accumulation across micro-batches free."*
- *"I verify a custom backward with `torch.autograd.gradcheck`: central finite differences in float64."*

## 🔗 Next
**Bridge (say it):** "We have gradients. A gradient says which way is uphill. So walk the other way,
a little, and repeat. That's training. Next we do exactly that with nothing but two tensors and a
for loop." → **M01E07: Linear regression with raw autograd.**

## 📹 Metadata
- **YouTube title:** `PyTorch autograd explained: the graph, backward, and why grads accumulate`
- **Thumbnail (3-5 words):** `WHY ZERO_GRAD?`
- **Short (which beat, 45-60 s):** call backward twice, grad goes 32 → 64. Then the "two paths add" line from the dry run. Caption: ".grad is a tip jar."
- **X post:** f(a,b) = (ab+b)². At a=3, b=2: autograd says 32 and 64, hand chain rule says 32 and 64, finite differences say 32.000000001 and 64.000000002. Then call backward again and you get 64 and 128, because grads add. That's why zero_grad exists. Code + video:
- **Sources:** [PyTorch: Automatic differentiation tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) · [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) · [torch.autograd.gradcheck](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html) · [Karpathy, micrograd lecture (Zero to Hero #1)](https://karpathy.ai/zero-to-hero.html)
