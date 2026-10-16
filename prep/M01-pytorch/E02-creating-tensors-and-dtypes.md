# EP004 · M01E02 · Creating tensors and dtypes

**Module:** PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 7-9 min (part 2 of "PyTorch in 1 Hour")
**Code:** `code/M01-pytorch/e02_creating_tensors_and_dtypes.py` · **Explainer:** `explainers/scenes/m01e02_creating_tensors_and_dtypes.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "I asked PyTorch to count to 1,000 in bfloat16. It got to 256 and stopped. Every LLM you've used
> was trained in this number format. Let's find out why it's still the right choice."

## ❓ The question (why this matters, 45 s)
Llama 3 was trained in bfloat16. DeepSeek-V3 went further and did most matmuls in FP8. Your
optimizer state is float32. Your token IDs are int64. Pick wrong and you get one of three failures:
an overflow to `inf` (fp16), a value that silently stops changing (bf16), or memory that's twice
what it needs to be (fp32 everywhere). This episode is the cheat sheet.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: 16 bits is a fixed budget for a price tag.** fp16 spends more digits after the decimal
> point: precise, but it can't print a price above $65,504. bf16 spends them on the size of the
> number: it can print a price of a trillion, but only to about 3 significant digits. In training,
> going off the edge of the tag (overflow) kills you. A slightly rounded price doesn't.

**The one picture: bit layouts.**

| dtype | sign | exponent | mantissa | max | eps (gap after 1.0) |
|---|---|---|---|---|---|
| float32 | 1 | 8 | 23 | 3.4e38 | 1.19e-7 |
| float16 | 1 | 5 | 10 | **65,504** | 9.8e-4 |
| bfloat16 | 1 | 8 | 7 | 3.4e38 | **7.8e-3** |

bf16 is just float32 with the bottom 16 mantissa bits chopped off. Same exponent, same range.

## 📐 The math (2 min)
Derive `torch.finfo` from the bit counts. `e` exponent bits, `m` mantissa bits:
```
bias = 2^(e-1) - 1                      # fp32/bf16: 127,  fp16: 15
eps  = 2^-m                             # spacing between 1.0 and the next number
max  = (2 - eps) * 2^bias               # top exponent, all mantissa bits set
tiny = 2^(1 - bias)                     # smallest normal number
```
fp16: `(2 - 2^-10) * 2^15 = 65504`. That's the whole overflow story.

**The spacing rule:** between `2^k` and `2^(k+1)`, numbers are `2^k * eps` apart. For bf16 at 256:
`256 * 2^-7 = 2`. Odd numbers above 256 don't exist in bf16.

### Dry run: counting in bf16
| c | c + 1 (exact) | nearest bf16 values | stored | why |
|---|---|---|---|---|
| 254 | 255 | 255 exists (spacing 1 below 256) | 255 | fine |
| 255 | 256 | 256 | 256 | fine |
| 256 | 257 | 256 or 258, tie | **256** | round half to even |
| 256 | 257 | ... | **256** | stuck forever |

Same thing with fp16 gets stuck at 2048 (spacing 2 from 2048 up). This is why **accumulators
(loss sums, optimizer state, master weights) stay in float32** even in "bf16 training."

## 💻 Code it (4 min)
**Creation tour, with the defaults that bite:**
```python
torch.zeros(2, 3).dtype            # float32
torch.arange(5).dtype              # int64  <- not float!
torch.arange(0, 1, 0.25)           # [0, .25, .5, .75]   end excluded
torch.linspace(0, 1, 5)            # [0, .25, .5, .75, 1] end included
torch.full((2,), 7).dtype          # int64, inferred from the 7
torch.tensor([1.0, 2]).dtype       # float32
(torch.tensor([1, 2]) / 2).dtype   # float32: / is true division
torch.zeros_like(x)                # same shape, dtype AND device as x
```
**The hand-derived finfo:**
```python
LAYOUT = {torch.float32: (8, 23), torch.float16: (5, 10), torch.bfloat16: (8, 7)}

def float_limits(exp_bits, man_bits):
    bias = 2 ** (exp_bits - 1) - 1
    eps = 2.0 ** -man_bits
    max_ = (2 - eps) * 2.0 ** bias
    tiny = 2.0 ** (1 - bias)
    return eps, max_, tiny
```
Then the three demos: `0.1 + 0.2`, overflow (`70000` in fp16 vs bf16), the bf16 counter.

## ✅ The receipt (1 min)
```python
for dt, (e, m) in LAYOUT.items():
    assert float_limits(e, m) == (torch.finfo(dt).eps, torch.finfo(dt).max, torch.finfo(dt).tiny)
```
Exact equality, not approx: these are powers of two, so the hand math is exact. Plus: fp16 70000
is `inf`, bf16 is finite; 257 → 256 in bf16 but stays 257 in fp16; the bf16 counter sticks at 256.
`pytest ... -q` → **5 passed** (measured on M4 Mac mini).

## 📊 Measure (1 min)
Measured on M4 Mac mini, actual script output:
```
python 0.1+0.2 == 0.3 ? False
ten 0.1s in fp32 = 1.0000001192  == 1 ? False
bf16 counting to 1000 gives: 256.0
fp16: tensor([60000.,    inf])    bf16: tensor([59904., 70144.])
257 in bf16: 256.0
1 + 1/512 in bf16 == 1 ? True
sum of squares of 4096 fours, fp32: 65536.0   fp16: inf
```
The last line is the realistic one: a norm over a 4096-wide hidden state (Llama 3 8B's width) with
modest activations overflows fp16. That's why fp16 training needed loss scaling and bf16 doesn't.

## ⚠️ Gotchas
- **Compare floats with `torch.isclose` / `assert_close`, never `==`.** Ten 0.1s is not 1.0.
- **`torch.arange(5)` is int64.** Multiply by a float tensor and it promotes, but
  `torch.arange(5) / 0` gives `[nan, inf, inf, inf, inf]` while `torch.arange(5) // 0` raises
  `RuntimeError: ZeroDivisionError`. Pass `dtype=torch.float32` when you mean floats.
- **`torch.arange(0, 1, 0.1)` length is fragile** with float steps. Use `linspace` when you need
  an exact count.
- **Embedding indices must be int64 (or int32).** Passing floats gives
  `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int`.
- **`torch.empty` is not zeros.** It's whatever was in memory.
- **No float64 on MPS** (`TypeError: Cannot convert a MPS Tensor to float64 dtype...`).
- **bf16 on CPU is very slow** (0.003 TFLOP/s matmul, measured in M00E02). Use it on the GPU.

## 🎤 Interview lines
- *"bf16 keeps float32's 8 exponent bits, so it has the same range and doesn't overflow where fp16
  does. It trades precision for that: about 3 significant digits."*
- *"That's why mixed-precision training keeps master weights, optimizer state and reductions in
  fp32. A bf16 accumulator stops changing once updates are smaller than its spacing."*
- *"fp16 tops out at 65,504, so fp16 training needs dynamic loss scaling. bf16 usually doesn't."*

## 🔗 Next
**Bridge (say it):** "We can make tensors and pick their number format. Now let's cut them up.
Slicing, transposing, reshaping, and the one rule for when PyTorch copies your data behind your
back." → **M01E03: Indexing, views and strides.**

## 📹 Metadata
- **YouTube title:** `bfloat16 vs float16 vs float32: the dtype cheat sheet for training LLMs`
- **Thumbnail (3-5 words):** `bf16 CAN'T COUNT`
- **Short (which beat, 45-60 s):** the counter. Loop to 1000 in bf16, prints 256. Explain the spacing in 20 s. Caption: "Your LLM's number format can't count past 256."
- **X post:** I asked PyTorch to count to 1000 in bfloat16 by adding 1. It stops at 256. Above 256, bf16 numbers are 2 apart, and 257 rounds back to 256. Same 16 bits as fp16, spent on range instead of precision. That's why optimizer state stays fp32. Code + video:
- **Sources:** [PyTorch docs: torch.finfo](https://docs.pytorch.org/docs/stable/type_info.html) · [Tensor creation ops](https://docs.pytorch.org/docs/stable/torch.html#creation-ops) · [Micikevicius et al. 2018, Mixed Precision Training](https://arxiv.org/abs/1710.03740)
