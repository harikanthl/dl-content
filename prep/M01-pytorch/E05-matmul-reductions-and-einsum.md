# EP007 · M01E05 · Matmul, reductions and einsum

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 7-9 min (part 5 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e05_matmul_reductions_and_einsum.py` · **Explainer:** `explainers/scenes/m01e05_matmul_reductions_and_einsum.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "Attention, the core of every LLM, is two matrix multiplies and a softmax. I'll write the first
> one three different ways, prove they're identical, and then match PyTorch's own attention kernel
> within float32 tolerance. In under nine minutes."

## ❓ The question (why this matters, 45 s)
Almost all the FLOPs in GPT-2 or Llama 3 are matmuls: QKV projections, attention scores, attention
output, the MLP. If you're unsure whether to use `*` or `@`, which axis `dim=-1` is, or why `bmm`
rejects your 4-D tensor, you can't read a single model file. And einsum is how you write the
shapes down so the next person (you, next month) doesn't have to guess.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: `*` is pairing socks, `@` is a survey.** `*` takes two equal piles and pairs item i with
> item i. `@` asks every row "how much do you agree with every column?" and writes down one score
> per pair. Attention scores are that survey: every query asks every key "how relevant are you?"

> **`dim` is the axis you squash.** `sum(dim=1)` on a `(2, 3)` grid squashes the 3 columns
> into one: you get one number per row, shape `(2,)`. The dim you name is the dim that disappears.

| op | inputs | what it does |
|---|---|---|
| `a * b` | broadcastable | elementwise |
| `a @ b` / `torch.matmul` | `(..., n, k) @ (..., k, m)` | matmul, batches over leading dims (broadcasts them) |
| `torch.mm` | exactly 2-D | matmul, no batching |
| `torch.bmm` | exactly 3-D, same batch | batched matmul |
| `torch.einsum` / `einops.einsum` | anything | you name every dim |

## 📐 The math (2 min)
Scaled dot-product attention for one head (Vaswani et al. 2017):
```
S = Q Kᵀ / √d          Q, K: (B, H, T, d)   ->  S: (B, H, T, T)
P = softmax(S, dim=-1)  each row (one query) sums to 1 over keys
O = P V                 (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)
```
Score entry: `S[b, h, q, k] = Σ_d Q[b, h, q, d] · K[b, h, k, d]`.

**einsum reads exactly like that sum:** `"bhqd,bhkd->bhqk"`. Rule: a letter on the left that's
missing on the right gets summed over. Here that's `d`.

### Dry run: reductions on `x = [[0, 1, 2], [3, 4, 5]]`, shape (2, 3)
| call | result | shape | what disappeared |
|---|---|---|---|
| `x.sum()` | 15 | () | everything |
| `x.sum(dim=0)` | [3, 5, 7] | (3,) | the rows |
| `x.sum(dim=1)` | [3, 12] | (2,) | the columns |
| `x.sum(dim=1, keepdim=True)` | [[3], [12]] | (2, 1) | columns, but kept as size 1 so it broadcasts back |
| `x.max(dim=1)` | values [2, 5], indices [2, 2] | (2,), (2,) | returns both |

## 💻 Code it (from scratch, 3 min)
```python
def scores_bmm(q, k):
    B, H, T, d = q.shape
    q3, k3 = q.reshape(B * H, T, d), k.reshape(B * H, T, d)   # bmm only takes 3-D
    s = torch.bmm(q3, k3.transpose(1, 2))                       # (BH, T, T)
    return s.view(B, H, T, T)

def scores_einsum(q, k):
    return torch.einsum("bhqd,bhkd->bhqk", q, k)

def scores_einops(q, k):
    return einops.einsum(q, k, "b h q d, b h k d -> b h q k")

def attention(q, k, v):
    s = scores_einsum(q, k) / math.sqrt(q.shape[-1])
    p = s.softmax(dim=-1)            # dim=-1: normalize over KEYS, one distribution per query
    return p @ v
```
Point at the docstring of each: the einsum string *is* the shape documentation. `bmm` needed two
reshapes and a transpose that you have to verify in your head.

## ✅ The receipt (1 min)
```python
ref = q @ k.transpose(-2, -1)
for fn in (scores_bmm, scores_einsum, scores_einops):
    torch.testing.assert_close(fn(q, k), ref)
torch.testing.assert_close(attention(q, k, v), F.scaled_dot_product_attention(q, k, v))
```
Our five-line attention matches PyTorch's production kernel. Plus a sanity test that `*` and `@`
give different answers on `[[1, 2], [3, 4]]`. `pytest ... -q` → **3 passed** (measured on M4 Mac
mini, MPS).

## 📊 Measure (1 min)
Measured on M4 Mac mini, MPS, GPT-2-small head shape: B=8, H=12, T=512, d=64.
```
scores_bmm       2.05 ms
scores_einsum    1.86 ms
scores_einops    1.85 ms
q @ k.mT         1.92 ms
my attention     7.64 ms
sdpa             2.52 ms
```
- The three score methods are **the same speed** (within noise). einsum and einops compile down to
  the same batched matmul. Readability is free.
- Full attention: ours **7.64 ms**, `F.scaled_dot_product_attention` **2.52 ms**, 3x faster. Ours
  writes the `(8, 12, 512, 512)` score matrix (100 MB in fp32) to memory, reads it back for softmax,
  writes it again, reads it again for `@ v`. The fused kernel avoids most of that traffic. That gap
  is the whole story of FlashAttention (M08).

## ⚠️ Gotchas
- **`*` when you meant `@`.** If shapes happen to broadcast (square matrices), there's no error.
  Just a wrong answer.
- **`softmax(dim=-2)` instead of `dim=-1`.** Normalizes over queries instead of keys. Every shape is
  still right. Only the receipt catches it.
- **`bmm` is strict:** `RuntimeError: batch1 must be a 3D tensor`. Use `@` or `einsum` for 4-D.
- **`mm` is strict too:** `RuntimeError: self must be a matrix`.
- **`.T` on >2-D tensors is deprecated.** Use `.mT` (transpose last two dims) or `transpose(-2, -1)`.
- **Forgetting `/ √d`.** With d=64, raw scores have std ≈ 8 instead of 1. Softmax saturates and
  gradients vanish. The receipt against SDPA catches it.
- **einsum letters are per call.** `"ij,jk->ik"` and `"ab,bc->ac"` are identical. Only the
  pattern matters.

## 🎤 Interview lines
- *"`*` is elementwise; `@` is matmul and broadcasts over leading batch dims. `bmm` is the strict
  3-D version."*
- *"The dim you pass to a reduction is the one that disappears. In attention, softmax is over the
  last dim, the keys, so each query's weights sum to one."*
- *"I write attention as `einsum('bhqd,bhkd->bhqk')` because the string documents the shapes, and it
  costs nothing: it lowers to the same batched matmul."*

## 🔗 Next
**Bridge (say it):** "We can now compute anything forward: matmuls, sums, attention. But training
needs the other direction, how much each input should change to lower the loss. PyTorch does that
for you automatically. Next we open up how." → **M01E06: Autograd, how PyTorch computes gradients.**

## 📹 Metadata
- **YouTube title:** `PyTorch matmul vs bmm vs einsum: attention scores 3 ways (all identical)`
- **Thumbnail (3-5 words):** `ATTENTION, 3 WAYS`
- **Short (which beat, 45-60 s):** the five-line attention matching `F.scaled_dot_product_attention`, then the 7.6 ms vs 2.5 ms timing. Caption: "Same answer. 3x slower. Why?"
- **X post:** Wrote attention scores three ways (bmm, torch.einsum, einops). Identical outputs, identical speed: 1.9 ms on M4 MPS at GPT-2 head shapes. Full 5-line attention matches F.scaled_dot_product_attention exactly, but SDPA is 3x faster: it doesn't round-trip the 100 MB score matrix. Code + video:
- **Sources:** [torch.einsum](https://docs.pytorch.org/docs/stable/generated/torch.einsum.html) · [torch.matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html) · [einops](https://einops.rocks) · [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) · [F.scaled_dot_product_attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
