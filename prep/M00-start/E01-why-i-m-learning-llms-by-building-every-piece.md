# EP001 · M00E01 · Why I'm learning LLMs by building every piece

**Module:** Start Here · **Hardware:** Mac · **Target length:** 10-12 min
**Code:** `code/M00-start/e01_why_i_m_learning_llms_by_building_every_piece.py` · **Explainer:** `explainers/scenes/m00e01_why_i_m_learning_llms_by_buildin.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "This is softmax. Three lines. It's in every LLM ever trained. And the version in most
> textbooks returns NaN the moment a model gets confident. I'm going to build every piece of an
> LLM like this, and check every single one against PyTorch, because 'it looks right' is how you
> ship that bug."

## ❓ The question (why this matters, 45 s)
You can call `model.generate()` without knowing anything. That works until it doesn't.
The loss goes NaN at step 40,000. Inference is 3x slower than the blog post. The interviewer
asks "why does RMSNorm not subtract the mean?" and you have nothing.

Every one of those is a question about a piece. GPT-2, Llama 3, DeepSeek-V3: they are all the
same dozen pieces in different arrangements. Tokenizer, embedding, attention, norm, MLP, loss,
optimizer, KV cache. If you've built each piece once with your own hands, the big models stop
being magic. They become a parts list.

That's the series. 216 episodes. Tensors to SGLang. Every piece built from scratch, and every
piece gets a receipt.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: learning to cook vs ordering takeout.** Takeout is faster tonight. But if the food is
> too salty you can't fix it, you can only complain. Cooking every dish once means that when a
> restaurant meal tastes off, you know it's the salt, not the oven.
>
> **The receipt is tasting the food.** A cook who never tastes is guessing. Every time we build
> something, we taste it against the reference: PyTorch's own version. Same input, same output,
> to the last few decimal places. If it doesn't match, we don't move on.

**The one picture (explainer):** the map. 18 modules as a staircase. Bottom step: tensors on a
Mac. Top step: serving a model with SGLang on a rented GPU. A small green checkmark (the receipt)
lights up on every step.

## 📐 The map (2-3 min, screen share of the printed table)
The code prints this from `curriculum/curriculum.json`. Talk through it in four chunks.

| Chunk | Modules | Episodes | Hardware |
|---|---|---|---|
| **Foundations** | M00 Start, M01 PyTorch, M02 Math, M03 Neural nets, M04 Tokenization | 44 | this Mac |
| **The model** | M05 Transformer layer by layer, M06 Training an LLM | 34 | Mac, then 15 min of 8xH100 |
| **Making it fast** | M07 GPUs/CUDA/Triton, M08 FlashAttention, M09 Modern architectures, M10 Distributed | 52 | RunPod 4090, then 4x GPU |
| **After training** | M11 Post-training, M12 Quantization, M13 Inference engines, M14 Evals, M15 DeepSeek, M16 Frontier, M17 Capstones | 86 | RunPod |

The line to say: **"The first 62 episodes need nothing but this Mac."** (M00 through M05.)
That matters. You don't need a GPU to start. You need to start.

**How to follow along:** one repo. Every episode has one Python file in `code/`, and that file
ends in a test. `pytest` passing is the receipt. Clone it, run the test, then try to write the
file yourself without looking.

## 💻 Code it (4 min)
```python
def softmax_naive(x, dim=-1):
    # textbook formula: exp(x) / sum(exp(x))
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)

def softmax_stable(x, dim=-1):
    # subtract the max first. softmax(x) == softmax(x - c) for any constant c,
    # so the answer is identical, but now the biggest exponent is exp(0) = 1.
    x = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)

def receipt(mine, ref, name):
    # the series' signature: never trust, always check against the reference
    try:
        torch.testing.assert_close(mine, ref)
        print(f"RECEIPT  {name:<16} PASS")
    except AssertionError as err:
        print(f"RECEIPT  {name:<16} FAIL  ({str(err).splitlines()[0]})")
```
Then in main: normal logits (`randn`), and the same logits times 100 (a model that got confident).

**The trap, named:** the naive softmax passes on normal inputs. You'd ship it. It only breaks on
big logits, and big logits are exactly what a trained model produces.

**Why the fix is correct (say this):** multiply top and bottom by `exp(-c)`:
`exp(x_i - c) / sum_j exp(x_j - c) = exp(x_i) exp(-c) / (exp(-c) sum_j exp(x_j))`. The `exp(-c)`
cancels. Pick `c = max(x)` and the largest exponent is `exp(0) = 1`. Nothing can overflow, and at
least one term in the sum is 1, so the denominator can't underflow to zero either.

### Dry run: why it breaks (float32 overflows past exp(88.7))
| logit | exp(logit), float32 | naive softmax | stable: logit - max | exp | stable softmax |
|---|---|---|---|---|---|
| 1000 | inf | inf/inf = **nan** | 0 | 1.0 | 0.731 |
| 999 | inf | inf/inf = **nan** | -1 | 0.368 | 0.269 |
| 0 | 1.0 | 1/inf = 0 | -1000 | 0.0 | 0.000 |

## ✅ The receipt (1 min)
Measured on M4 Mac mini (torch 2.14.1, MPS):
```
RECEIPT  naive, small     PASS
RECEIPT  naive, big       FAIL  (Tensor-likes are not close!)
RECEIPT  stable, big      PASS
naive big has nan: True
```
`pytest code/M00-start/e01_why_i_m_learning_llms_by_building_every_piece.py -q` → **3 passed**.
The tests check: the map has 18 modules summing to 216 episodes, the naive version really does
NaN on `[1000, 999, 0]`, and the stable version matches `torch.softmax` at scales 1, 100 and 10,000.

## 📊 Measure (1 min)
One number: the naive softmax survives logits up to about **88.7** in float32 (`log` of the
float32 max, 3.4e38). A trained LLM's logits blow past that easily, and in bfloat16 attention
scores it's the same ceiling. So the textbook version doesn't fail rarely. It fails on exactly the
inputs you care about.

## ⚠️ Gotchas
- **"It passed my test" is not a receipt** if the test only used small random inputs. The first
  receipt passed. Test at the scales the real model sees.
- `assert_close` picks tolerances from the dtype (float32: rtol 1.3e-6, atol 1e-5). Don't pass
  huge `atol` to make a failure go away. That's not a receipt, that's a cover-up.
- NaN != NaN. `assert_close` treats a NaN anywhere as a failure by default, which is what you want.
- This is also why `torch.softmax` and `F.cross_entropy` exist as fused ops. You'll see the same
  max-subtraction trick again inside FlashAttention's online softmax (M08).

## 🎤 Interview lines
- *"Softmax is shift-invariant, so we subtract the row max before exponentiating. Same answer,
  no overflow."*
- *"I verify every component against a reference implementation with `assert_close`, at realistic
  input scales, before I build on top of it."*
- *"Most training instabilities I've seen are numerics: an exp, a log or a division somewhere
  that's fine on random inputs and not on trained activations."*

## 🔗 Next
Next is **M00E02: Setup.** uv, PyTorch, MPS vs CUDA, and how to rent an H100 for 15 minutes
without waking up to a $400 bill. Then M01 starts the real work: what a tensor actually is.

## 📹 Metadata
- **YouTube title:** `Building an LLM from scratch, every piece, with receipts (216 episodes)`
- **Thumbnail (3-5 words):** `EVERY PIECE. CHECKED.`
- **Short (which beat, 45-60 s):** the softmax receipt. Naive passes, x100 fails with NaN, fix, passes. Caption: "The textbook softmax is broken."
- **X post:** Starting a 216-episode series building an LLM from first principles. Rule #1: every piece gets checked against PyTorch. Episode 1's receipt already caught a bug: the textbook softmax returns NaN once logits pass ~88. Code + video:
- **Sources:** [Stanford CS336](https://cs336.stanford.edu) · [Karpathy, Zero to Hero](https://karpathy.ai/zero-to-hero.html) · [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html)

> Note: the curriculum's `learn` field for this episode says "186 videos". The curriculum now has
> **216**. Say 216 on camera.
