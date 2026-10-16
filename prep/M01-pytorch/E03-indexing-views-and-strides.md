# EP005 · M01E03 · Indexing, views and strides

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 7-9 min (part 3 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e03_indexing_views_and_strides.py` · **Explainer:** `explainers/scenes/m01e03_indexing_views_and_strides.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "I changed one number in a column of a tensor, and a different tensor changed too. That's not a
> bug. That's a view. Half of PyTorch is views, and once you can predict which ops copy, you stop
> getting surprised by both your memory and your data."

## ❓ The question (why this matters, 45 s)
Every attention layer does this: `x.view(B, T, n_heads, head_dim).transpose(1, 2)`. Then later
`y.transpose(1, 2).contiguous().view(B, T, C)`. That `.contiguous()` is in nanoGPT, in Llama's
reference code, everywhere. Remove it and you get a RuntimeError. Today you'll know exactly why it's
there, and you'll be able to say whether it costs anything.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a view is a window, a copy is a photo.** Look at the shelf through a window (a slice) and
> someone moves a book, you see it move. Take a photo (a copy) and it never changes. Slicing gives
> you a window. Fancy indexing gives you a photo.

From M01E01: tensor = storage + shape + stride + offset. **A view is any op you can express by
changing only shape, stride and offset.** If you can't, PyTorch has to copy.

| Op | View or copy | How |
|---|---|---|
| `t[1]`, `t[:, 1]`, `t[::2]` | view | new offset / stride |
| `t.T`, `t.transpose(0, 1)`, `t.permute(...)` | view | strides reordered |
| `t.view(...)` | view, **or error** | needs compatible strides |
| `t.reshape(...)` | view if possible, **else copy** | silent |
| `t.contiguous()` | returns `t` itself if already contiguous, else copy | |
| `t[[0, 2]]`, `t[t > 5]` | **always copy** | rows aren't evenly spaced |

## 📐 The math (2 min)
Start from `t = torch.arange(12.).view(3, 4)`: shape `(3, 4)`, stride `(4, 1)`.

### Dry run: what each slice does to the recipe
| expression | shape | stride | offset | contiguous | reason |
|---|---|---|---|---|---|
| `t` | (3, 4) | (4, 1) | 0 | yes | |
| `t[1]` | (4,) | (1,) | **4** | yes | skip one row = 4 elements |
| `t[:, 1]` | (3,) | **(4,)** | 1 | no | walk down: jump a whole row each step |
| `t[::2]` | (2, 4) | **(8, 1)** | 0 | no | every other row: double the row stride |
| `t.T` | (4, 3) | **(1, 4)** | 0 | no | strides swapped |

**Contiguous** means: the strides are exactly `contiguous_strides(shape)` from M01E01. Reading the
elements in order walks memory left to right with no gaps.

**Why `t.T.view(12)` fails (the correctness argument):** a 1-D view of 12 elements needs a single
stride `s` so that element `k` lives at `k * s`. In `t.T`, logical order is
`0, 4, 8, 1, 5, 9, 2, ...`. The gaps are `+4, +4, -7, +4, ...`. No single stride produces that.
So no view exists, and PyTorch refuses rather than secretly copying. `reshape` copies instead.

## 💻 Code it (from scratch, 3 min)
Rebuild three views with only `as_strided` (size, stride, offset over the same storage):
```python
def my_transpose(t):
    return t.as_strided(size=t.shape[::-1], stride=t.stride()[::-1],
                        storage_offset=t.storage_offset())

def my_column(t, j):
    # column j: walk down rows (stride = row stride), start j elements in
    return t.as_strided(size=(t.shape[0],), stride=(t.stride(0),),
                        storage_offset=t.storage_offset() + j)

def my_every_other_row(t):
    return t.as_strided(size=((t.shape[0] + 1) // 2, t.shape[1]),
                        stride=(2 * t.stride(0), t.stride(1)),
                        storage_offset=t.storage_offset())

def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
```
Then on camera: write through `col = t[:, 1]` and show `t` changed. Trigger the `view` error.
Show `reshape` works but `same_storage` is False.

## ✅ The receipt (1 min)
```python
for mine, ref in [(my_transpose(t), t.T), (my_column(t, 1), t[:, 1]),
                  (my_every_other_row(t), t[::2])]:
    torch.testing.assert_close(mine, ref)
    assert mine.stride() == ref.stride()
    assert same_storage(mine, t) and same_storage(ref, t)
```
Values match, strides match, and both share the original storage. Runs on MPS. Plus the
view-vs-reshape test and the what-copies test. `pytest ... -q` → **3 passed** (measured on M4 Mac mini).

## 📊 Measure (1 min)
Measured on M4 Mac mini, actual script output:
```
t              shape=(3, 4)    stride=(4, 1)   offset=0 contiguous=True
t[1]           shape=(4,)      stride=(1,)     offset=4 contiguous=True
t[:, 1]        shape=(3,)      stride=(4,)     offset=1 contiguous=False
t[::2]         shape=(2, 4)    stride=(8, 1)   offset=0 contiguous=False
t.T            shape=(4, 3)    stride=(1, 4)   offset=0 contiguous=False
t[0, 1] after writing col: 100.0
view fails: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces)
reshape shares storage? False
contiguous() on contiguous is free? True
fancy index t[[0, 2]] shares storage? False
mask t[t > 5] shares storage? False
```

## ⚠️ Gotchas
- **The error message:** `RuntimeError: view size is not compatible with input tensor's size and
  stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.`
  It appears right after a `transpose` or `permute`. Fix: `.contiguous().view(...)` or `.reshape(...)`.
- **`reshape` hides copies.** Fine for correctness, but in a hot loop it can double memory traffic.
  Use `view` when you expect no copy; the error is a free assertion.
- **In-place writes through views change the original.** `col[0] = 100` changed `t[0, 1]`. Want
  independence? `.clone()`.
- **`t[[0, 2]] = 0` does write into `t`** (that's `__setitem__`), but `u = t[[0, 2]]; u[:] = 0` does
  not. Reading with fancy indexing copies; assigning with it writes in place.
- **Autograd and in-place on views:** modifying a view of a leaf that requires grad raises
  `RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.`
  More in M01E06.

## 🎤 Interview lines
- *"Slicing and transpose return views: same storage, new strides. Advanced indexing and boolean
  masks return copies because the selected elements aren't evenly spaced."*
- *"`.view` never copies and errors if it can't. `.reshape` copies when it has to. After a transpose
  you usually need `.contiguous()` before `.view`."*
- *"In multi-head attention, the `transpose(1, 2).contiguous().view(B, T, C)` at the end is a real
  copy. It's required because heads are no longer adjacent in memory."*

## 🔗 Next
**Bridge (say it):** "Views let one tensor wear many shapes. Next, two tensors with *different*
shapes meet in one operation, and PyTorch quietly stretches one of them. That's broadcasting, and
it's the biggest source of silent bugs in this whole series." → **M01E04: Broadcasting, the bug factory.**

## 📹 Metadata
- **YouTube title:** `PyTorch views vs copies: strides, .view vs .reshape, and .contiguous()`
- **Thumbnail (3-5 words):** `WHEN DOES IT COPY?`
- **Short (which beat, 45-60 s):** write to `t[:, 1]`, the original changes; then `t[[0,2]]` doesn't. Caption: "Window or photo?"
- **X post:** Why does every attention implementation call .contiguous() before .view()? After transpose, no single stride walks the elements in order (gaps go +4, +4, -7...), so a view can't exist. PyTorch errors instead of secretly copying. Code + video:
- **Sources:** [PyTorch docs: Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html) · [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) · [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape.html)
