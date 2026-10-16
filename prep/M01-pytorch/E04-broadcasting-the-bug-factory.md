# EP006 · M01E04 · Broadcasting: the bug factory

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 7-9 min (part 4 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e04_broadcasting_the_bug_factory.py` · **Explainer:** `explainers/scenes/m01e04_broadcasting_the_bug_factory.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "My model was off by 0.1 on every prediction. The loss said 3.3. The right answer is 0.01. No
> error, no crash, just a number 300 times too big. One missing dimension did that, and today you'll
> learn to spot it in two seconds."

## ❓ The question (why this matters, 45 s)
Broadcasting is why you can write `x - x.mean(-1, keepdim=True)` in LayerNorm, add a `(C,)` bias
to a `(B, T, C)` activation, or apply a `(T, T)` causal mask to `(B, H, T, T)` attention scores.
It's everywhere in GPT-2. It's also the one PyTorch feature that turns a shape mistake into a
wrong number instead of an error. You need the two rules memorized.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a stamp and a sheet of paper.** You have a row of 4 colors (a stamp) and a sheet of 3 rows.
> Broadcasting stamps the same row onto every line of the sheet. It doesn't print 3 copies of the
> stamp first; it just reuses it. A size-1 dimension means "this is a stamp, reuse me."

**The two rules. Write them on screen:**
1. **Line the shapes up from the right.** Missing dims on the left count as size 1.
2. **Each pair must be equal, or one of them must be 1.** The 1 gets stretched. Otherwise: error.

## 📐 The math (2 min)
### Dry run: `(8, 1, 6, 1)` with `(7, 1, 5)`
| dim (from right) | a | b | rule | result |
|---|---|---|---|---|
| -1 | 1 | 5 | one is 1 | 5 |
| -2 | 6 | 1 | one is 1 | 6 |
| -3 | 1 | 7 | one is 1 | 7 |
| -4 | 8 | (missing → 1) | rule 1, then one is 1 | 8 |

Result: `(8, 7, 6, 5)`.

### Dry run: the bug, `(5, 1)` minus `(5,)`
| dim | pred `(5, 1)` | y `(5,)` → `(1, 5)` | result |
|---|---|---|---|
| -1 | 1 | 5 | 5 |
| -2 | 5 | 1 (padded) | 5 |

`(pred - y)` is **(5, 5)**, entry `[i, j] = pred_i - y_j`. You compared every prediction to every
target. The mean of its square is exactly `0.1² + 2·var(y)` (the cross terms cancel), not `0.1²`.
With these 5 random targets: `0.01 + 3.3153 = 3.3253`, which is what the code prints. The bigger
the spread of your labels, the more wrong the loss. Nothing about it looks broken.

**Pairwise distance, the useful side of the same mechanism:**
```
X: (N, D) -> X[:, None, :]  (N, 1, D)
Y: (M, D) -> Y[None, :, :]  (1, M, D)
diff:                       (N, M, D)   -> square, sum over D, sqrt -> (N, M)
```
The memory-cheap version expands the square:
```
||x - y||² = ||x||² + ||y||² - 2 x·y        ->   (N,1) + (1,M) - 2 X @ Y.T
```

## 💻 Code it (from scratch, 3 min)
```python
def broadcast_shape(a, b):
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1               # rule 1: pad on the left with 1
        y = b[-i] if i <= len(b) else 1
        if x != y and 1 not in (x, y):                # rule 2
            raise ValueError(f"cannot broadcast {a} and {b} at dim -{i}: {x} vs {y}")
        out.append(max(x, y))
    return tuple(reversed(out))

def pairwise_dist(X, Y):
    diff = X[:, None, :] - Y[None, :, :]              # (N, M, D)
    return diff.pow(2).sum(-1).sqrt()                 # (N, M)

def pairwise_dist_expanded(X, Y):
    sq = (X * X).sum(-1)[:, None] + (Y * Y).sum(-1)[None, :] - 2 * X @ Y.T
    return sq.clamp_min(0).sqrt()                     # rounding can dip below 0

def the_bug(n=5):
    y = torch.randn(n)                  # targets, (N,)
    pred = y[:, None] + 0.1             # predictions, (N, 1). off by exactly 0.1
    wrong = ((pred - y) ** 2).mean()    # (N, N)!
    right = ((pred.squeeze(-1) - y) ** 2).mean()
```

## ✅ The receipt (1 min)
```python
for a, b in cases:
    assert broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))
torch.testing.assert_close(pairwise_dist(X, Y), torch.cdist(X, Y))
torch.testing.assert_close(pairwise_dist_expanded(X, Y), torch.cdist(X, Y), atol=1e-3, rtol=1e-3)
```
Plus: incompatible shapes raise, and the bug test asserts the shape really is `(5, 5)`, the right
MSE is exactly 0.01, and the wrong one is over 10x larger. `pytest ... -q` → **3 passed** (measured
on M4 Mac mini, MPS).

## 📊 Measure (1 min)
Measured on M4 Mac mini, MPS, N = M = 1024, D = 128:
```
(pred - y).shape = (5, 5)   wrong MSE = 3.3253   right MSE = 0.0100
broadcast     24.10 ms  (builds 1024x1024x128 = 512 MB)
expanded       0.59 ms
torch.cdist    0.32 ms
```
The broadcast version is the clearest code and **40x slower**, because it materializes a 512 MB
intermediate. The expanded version turns the work into one matmul. (`torch.cdist` uses the same
matmul trick internally for inputs this size.) Broadcasting is free when it stretches the *output*;
it's expensive when it inflates an *intermediate*.

## ⚠️ Gotchas
- **The silent one: `(N,)` vs `(N, 1)`.** Model outputs `(N, 1)`, labels are `(N,)`. Fix with
  `pred.squeeze(-1)` or `y.view(-1, 1)`, and add `assert pred.shape == y.shape` before every loss.
- **PyTorch warns you, once.** `F.mse_loss` prints `UserWarning: Using a target size
  (torch.Size([5])) that is different to the input size (torch.Size([5, 1])). This will likely lead
  to incorrect results due to broadcasting.` Hand-written losses don't warn at all.
- **Forgetting `keepdim=True`.** `x - x.mean(-1)` on `(B, T, C)` gives a `(B, T)` mean that lines up
  against `C`. If `T == C` it runs and is wrong; otherwise it errors. Always `keepdim=True`.
- **The loud one is the good one:** `RuntimeError: The size of tensor a (3) must match the size of
  tensor b (4) at non-singleton dimension 1`. Be grateful for it.
- **In-place ops don't broadcast the left side.** `a += b` can't grow `a`: `(3,) += (2, 3)` raises
  `output with shape [3] doesn't match the broadcast shape [2, 3]`.
- **Expanded distance is less accurate** (needs `atol=1e-3`): it subtracts large similar numbers.
  `clamp_min(0)` before `sqrt`, or you get NaN on the diagonal.

## 🎤 Interview lines
- *"Broadcasting aligns shapes from the right; each dim must match or be 1, and size-1 dims are
  stretched without copying."*
- *"The classic bug is subtracting an `(N,)` target from an `(N, 1)` prediction. You get an `(N, N)`
  matrix and a loss that's wrong but finite. I assert shapes before every loss."*
- *"For pairwise distances I use `‖x‖² + ‖y‖² − 2x·y` so the work is one matmul and I never build the
  `(N, M, D)` tensor."*

## 🔗 Next
**Bridge (say it):** "That `X @ Y.T` did the heavy lifting in our distance matrix. Next: matmul
properly. `@` vs `*` vs `mm` vs `bmm`, what the `dim` argument really means, and einsum, which lets
you write the shapes down instead of keeping them in your head." → **M01E05: Matmul, reductions and einsum.**

## 📹 Metadata
- **YouTube title:** `PyTorch broadcasting: the 2 rules and the silent (N,) vs (N,1) bug`
- **Thumbnail (3-5 words):** `(N,) ≠ (N,1)`
- **Short (which beat, 45-60 s):** the bug. Off by 0.1, loss says 3.33, shape is (5, 5). One `.squeeze(-1)` later: 0.01. Caption: "No error. Wrong answer."
- **X post:** Model off by exactly 0.1 everywhere. MSE should be 0.01. Mine said 3.33. Prediction was (N,1), target was (N,), so broadcasting made an (N,N) matrix of every prediction minus every target. No error. Assert shapes before every loss. Code + video:
- **Sources:** [PyTorch docs: Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html) · [torch.broadcast_shapes](https://docs.pytorch.org/docs/stable/generated/torch.broadcast_shapes.html) · [torch.cdist](https://docs.pytorch.org/docs/stable/generated/torch.cdist.html)
