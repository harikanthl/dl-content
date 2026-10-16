# EP003 · M01E01 · Tensors: what they actually are

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 6-8 min (part 1 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e01_tensors_what_they_actually_are.py` · **Explainer:** `explainers/scenes/m01e01_tensors_what_they_actually_are.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "A 2D tensor isn't 2D. Nothing in memory is. It's a flat list of numbers plus a tiny recipe that
> says how to read it as a grid. Once you see the recipe, transpose, slicing and half of PyTorch's
> error messages make sense."

## ❓ The question (why this matters, 45 s)
GPT-2 small's token embedding is a `(50257, 768)` tensor. Llama 3 8B's attention reshapes
activations from `(B, T, 4096)` to `(B, 32, T, 128)` and back on every layer. If you think of
these as "grids," you'll be confused when `.view()` throws, when a transpose costs nothing, or when
your memory doubles because something quietly copied. Think of them as storage plus a recipe and
none of that is surprising.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a tensor is a long bookshelf plus a reading rule.** The books are in one long row. The
> rule says "a new row starts every 3 books." Change the rule to "read down instead of across" and
> you've transposed the shelf without moving a single book.

**The five parts.** Say them, write them on screen:

| Part | What it is | `t = torch.tensor([1.,2,3,4,5,6]).view(2,3)` |
|---|---|---|
| **storage** | flat block of bytes | 6 floats, 24 bytes |
| **shape** | how many along each dim | `(2, 3)` |
| **stride** | how many elements to jump per step in each dim | `(3, 1)` |
| **dtype** | how to read the bytes | `float32`, 4 bytes each |
| **device** | where the bytes live | `cpu` or `mps:0` |

Plus a sixth, small one: **storage offset**, where in the storage this tensor starts.

## 📐 The math (1-2 min)
One formula. This formula IS a tensor:

```
flat_index = offset + i_0 * stride_0 + i_1 * stride_1 + ... + i_k * stride_k
```

Contiguous (row-major) strides: the last dim moves by 1, each earlier dim jumps over everything
after it.
```
stride_k = shape_{k+1} * shape_{k+2} * ... * shape_last        (stride_last = 1)
(2, 3)     -> (3, 1)
(2, 3, 4)  -> (12, 4, 1)
```

### Dry run: storage `[1, 2, 3, 4, 5, 6]`
| index | t: shape (2,3), stride (3,1) | value | t.T: shape (3,2), stride (1,3) | value |
|---|---|---|---|---|
| (0,0) | 0·3 + 0·1 = 0 | 1 | 0·1 + 0·3 = 0 | 1 |
| (0,1) | 0·3 + 1·1 = 1 | 2 | 0·1 + 1·3 = 3 | 4 |
| (0,2) | 0·3 + 2·1 = 2 | 3 | n/a | |
| (1,0) | 1·3 + 0·1 = 3 | 4 | 1·1 + 0·3 = 1 | 2 |
| (1,1) | 1·3 + 1·1 = 4 | 5 | 1·1 + 1·3 = 4 | 5 |
| (2,1) | n/a | | 2·1 + 1·3 = 5 | 6 |

Same six numbers. Different stride. That's a transpose.

## 💻 Code it (from scratch, 3 min)
```python
def contiguous_strides(shape):
    # row-major: last dim moves by 1, each earlier dim jumps over everything after it
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


class HandTensor:
    """Storage + shape + stride + offset. No copying, ever."""
    def __init__(self, storage, shape, stride=None, offset=0):
        self.storage = storage                          # flat python list
        self.shape = tuple(shape)
        self.stride = tuple(stride) if stride else contiguous_strides(shape)
        self.offset = offset

    def __getitem__(self, idx):
        # the one formula that IS a tensor
        flat = self.offset + sum(i * s for i, s in zip(idx, self.stride))
        return self.storage[flat]
```
Then on the real thing: `t.stride()`, `t.storage_offset()`, `t.untyped_storage().nbytes()`,
`t.data_ptr()`. Show `t.T` has the **same `data_ptr`** and swapped strides.

## ✅ The receipt (1 min)
```python
assert HandTensor(flat, (2, 3)).tolist() == t.tolist()
assert HandTensor(flat, (3, 2), stride=(1, 3)).tolist() == t.T.tolist()
s = torch.as_strided(torch.tensor(flat), size=(3, 2), stride=(1, 3))
torch.testing.assert_close(s, t.T)
```
Plus `contiguous_strides` matches `torch.empty(shape).stride()` on four shapes, and `t[1]` is the
same storage with offset 3. `pytest ... -q` → **4 passed** (measured on M4 Mac mini).

## 📊 Measure (1 min)
Measured on M4 Mac mini, actual script output:
```
t   : shape=(2, 3) stride=(3, 1) dtype=torch.float32 device=cpu   storage_bytes=24 ptr=0x84bad4c00
t.T : shape=(3, 2) stride=(1, 3) dtype=torch.float32 device=cpu   storage_bytes=24 ptr=0x84bad4c00
bf16: shape=(2, 3) stride=(3, 1) dtype=torch.bfloat16 device=cpu  storage_bytes=12 ptr=0x84bad4c40
mps : shape=(2, 3) stride=(3, 1) dtype=torch.float32 device=mps:0 storage_bytes=24 ptr=0x84bb00fc0
```
Read it out: transpose, same pointer, zero bytes copied. Change dtype, half the bytes, new pointer.
Change device, new pointer, same bytes. (The pointer values change every run. The pattern doesn't.)

## ⚠️ Gotchas
- **`.storage()` is deprecated.** It returns a `TypedStorage` and warns. Use `.untyped_storage()`.
- **Compare storages, not tensors, to detect sharing.** `t[1].data_ptr()` differs from
  `t.data_ptr()` (it points 12 bytes in), even though they share memory. Compare
  `x.untyped_storage().data_ptr()`.
- **`.to(dtype)` and `.to(device)` copy.** Only shape/stride tricks are free.
- **A tensor of 6 numbers can sit on a storage of a million.** Slice one row out of a huge tensor and
  keep it, and the whole storage stays alive. `.clone()` if you mean to let the big one go.

## 🎤 Interview lines
- *"A tensor is a view onto a flat storage, described by shape, stride, offset, dtype and device."*
- *"Transpose is free. It swaps strides and touches no data. That's why it returns a
  non-contiguous tensor."*
- *"Element (i, j) lives at offset + i·stride0 + j·stride1. Every view op is just a new choice of
  those numbers."*

## 🔗 Next
**Bridge (say it):** "So a tensor is bytes plus a recipe. Next: how you make one, and why the
`dtype` part of the recipe decides whether your model trains or quietly counts to 256 and stops."
→ **M01E02: Creating tensors and dtypes.**

## 📹 Metadata
- **YouTube title:** `What a PyTorch tensor actually is: storage, shape and stride`
- **Thumbnail (3-5 words):** `2D IS A LIE`
- **Short (which beat, 45-60 s):** the dry-run table. Same 6 numbers, change the stride, it's transposed. Caption: "Transpose moves zero bytes."
- **X post:** A PyTorch tensor is a flat block of bytes + shape + stride + offset + dtype + device. t.T has the same data_ptr as t: transpose swaps two numbers in the recipe and copies nothing. Built one by hand from a Python list. Code + video:
- **Sources:** [PyTorch: Tensors tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) · [torch.Tensor.stride](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html) · [torch.as_strided](https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html) · [Tensor views](https://docs.pytorch.org/docs/stable/tensor_view.html)
