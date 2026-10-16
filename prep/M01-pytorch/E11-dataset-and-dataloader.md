# EP013 · M01E11 · Dataset and DataLoader

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e11_dataset_and_dataloader.py` · **Explainer:** `explainers/scenes/m01e11_dataset_and_dataloader.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "A Dataset is two methods. A DataLoader is a for-loop with opinions. And the moment your
> examples have different lengths, the default DataLoader crashes. Let's fix that the way
> every LLM pipeline does."

## ❓ The question (why this matters, 45 s)
Images are all 28x28. Sentences are not. "hi" is 2 characters, the next line is 38.
A batch has to be one rectangular tensor `(B, T)`, so something has to decide what
happens to the short ones. That something is `collate_fn`, and it's the piece most
tutorials skip.

Every LLM we train later, the bigram model in M03, nanoGPT in M06, the SFT run in M11,
feeds data through exactly this shape: a Dataset that returns token ids, a collate that
pads and returns a mask. Today it's characters. Later it's BPE tokens (M04). Same code.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a library and a librarian.**
> The **Dataset** is the library. It answers two questions: "how many books?"
> (`__len__`) and "give me book number i" (`__getitem__`). That's all it does. It doesn't
> know about batches or shuffling.
>
> The **DataLoader** is the librarian. It shuffles the call numbers (the sampler), grabs
> books in piles of B (the batch sampler), and hands each pile to the packer.
>
> The **collate_fn** is the packer. Books come in different sizes, the box is rectangular.
> The default packer tries to stack them and gives up if they don't match. Our packer adds
> bubble wrap (PAD tokens) to the short ones and writes a note saying which parts are
> real book (the mask).

The picture: three rows of different lengths, then the same three rows right-padded with
grey PAD squares into a rectangle, with the boolean mask drawn underneath.

## 🪤 The trap
**The default collate calls `torch.stack`.** Ragged examples fail with:
```
RuntimeError: stack expects each tensor to be equal size, but got [11] at entry 0 and [10] at entry 1
```
Run it on camera first (`test_default_collate_fails_on_ragged` receipts the message), then
fix it.

**The second trap is macOS-specific.** With `num_workers > 0`, macOS starts workers with
`spawn`: each worker re-imports your script. No `if __name__ == "__main__":` guard and the
workers try to start their own DataLoaders. You get a wall of
`RuntimeError: An attempt has been made to start a new process before the current process
has finished its bootstrapping phase`. The guard is not optional on a Mac. Also: your
Dataset gets pickled to each worker, so no lambdas or open file handles as attributes.

## 📐 The math (derive on screen, 2–3 min)
Padding, in shapes:
1. Batch of `B` examples with lengths `L = [l₁ ... l_B]`. `T = max(L)`.
2. `out = full((B, T), PAD)`, then `out[b, :l_b] = x_b`. Right-padding: real tokens first.
3. Mask by broadcasting (E04 pays off):
   `arange(T)[None, :]` is `(1, T)`, `lengths[:, None]` is `(B, 1)`,
   `<` broadcasts to `(B, T)`. `mask[b, t] = t < l_b`. True on real tokens.
4. Waste: `1 − Σ l_b / (B·T)`. Our first sample batch, lengths `[9, 2, 10]` padded to 10:
   21 real out of 30, 30% padding. One long outlier per batch makes everything else pay.
   (That's why real pipelines bucket by length, or, like nanoGPT, skip padding entirely by
   cutting a long token stream into fixed-length chunks. M06.)

Why id 0 for PAD: the mask is what matters, but reserving 0 makes padding visible when
you print a batch, and `nn.Embedding(padding_idx=0)` can use it later.

## 💻 Code it (from scratch, 4–5 min)
```python
class LineDataset(Dataset):
    def __init__(self, path):
        self.lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
        chars = sorted(set("".join(self.lines)))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}     # 0 is PAD
        self.itos = {i: c for c, i in self.stoi.items()}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):                                    # one example, any length
        return torch.tensor([self.stoi[c] for c in self.lines[i]], dtype=torch.long)

def pad_collate(batch):                                          # list of (l_b,) tensors
    lengths = torch.tensor([len(x) for x in batch])              # (B,)
    T = int(lengths.max())
    out = torch.full((len(batch), T), PAD, dtype=torch.long)     # (B, T)
    for b, x in enumerate(batch):
        out[b, :len(x)] = x
    mask = torch.arange(T)[None, :] < lengths[:, None]           # (B, T)
    return out, lengths, mask

g = torch.Generator().manual_seed(0)
dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=pad_collate, generator=g)
```
On camera, print each batch decoded. Measured output:
```
(3, 10) [9, 2, 10]  ['short one', 'hi', 'on the mat']
(3, 38) [2, 38, 35] ['ok', 'a much longer line about dogs and cats', 'transformers eat padded batches too']
(2, 20) [20, 11]    ['pytorch pads batches', 'the cat sat']
```
Point at the last batch: 8 lines, batch 3, so the last batch has 2. `drop_last=True` if
you need fixed B (you will, for `torch.compile` in M06).

Note the vocabulary: 21 distinct characters + PAD = 22. `ds[0]` for "the cat sat" is
`[19, 9, 6, 1, 4, 2, 19, 1, 18, 2, 19]`. Space is id 1, it sorts first.

## ✅ The receipt (1 min)
- `test_len_and_getitem`: 8 lines, `decode(ds[0]) == "the cat sat"`, no PAD in real data.
- `test_pad_collate_shapes_and_mask`: lengths `[2, 11, 38]`, shape `(3, 38)`,
  `mask.sum() == 51`, every masked-out position is PAD and every masked-in one isn't.
- `test_default_collate_fails_on_ragged`: the exact `stack expects each tensor...` error.
- `test_shuffle_is_reproducible_with_generator`: same seed same order, different seed
  different order, and every line appears exactly once per epoch.

`pytest -q` → **4 passed**.

## 📊 Measure (1 min)
Measured on M4 Mac mini: 40,000 lines, batch 64, 3 epochs, `persistent_workers=True`:

| num_workers | time |
|---|---|
| 0 | **0.33 s** |
| 2 | 1.22 s |
| 4 | 1.04 s |

Workers made it **3–4x slower.** Each worker is a separate process; batches come back
through inter-process pickling. When `__getitem__` is a dictionary lookup, that overhead is
bigger than the work. Workers pay off when `__getitem__` is expensive: decoding a JPEG,
reading from disk, running a tokenizer. Rule: start at 0, measure, then add workers.
(On MNIST in E12 we skip the DataLoader entirely and it's 2x faster on MPS.)

## ⚠️ Gotchas
- **Ragged + default collate** → `stack expects each tensor to be equal size`.
- **No `__main__` guard with workers on macOS** → bootstrapping-phase RuntimeError.
- **Shuffling without a generator.** Uses the global RNG, so adding a dropout layer
  changes your data order. Give the DataLoader its own `torch.Generator` (and save its
  state in E13).
- **Workers and randomness.** Each worker reseeds torch, Python `random` and global
  NumPy (NumPy since PyTorch 1.9; old blog posts say otherwise). What does NOT get
  reseeded: an RNG object you built in `__init__`, like `self.rng = np.random.default_rng(0)`.
  It's pickled into every worker, so all workers produce the same "random" augmentations.
  Reseed it in `worker_init_fn`.
- **Doing heavy work in `__init__` vs `__getitem__`.** Loading everything in `__init__` is
  fine until it doesn't fit in RAM. Then you need lazy loading or memory-mapping (M04's
  token `.bin` files).
- **Building the vocab from val data.** Here it's one file so it doesn't matter. Later,
  build vocab on train only, or you'll hit a `KeyError` on unseen characters, or worse,
  leak.
- **`shuffle=True` on the validation loader.** Harmless for accuracy, but makes logged
  per-batch numbers unrepeatable. Shuffle train only.

## 🎤 Interview lines
- *"A map-style Dataset implements __len__ and __getitem__. The DataLoader adds sampling,
  batching, collation and optional multiprocessing on top."*
- *"For variable-length sequences I write a collate_fn that right-pads to the batch max and
  returns a boolean mask, and the model uses the mask in attention and in the loss."*
- *"num_workers isn't free: workers are processes and batches cross a process boundary. If
  __getitem__ is cheap, zero workers is often fastest. I measure it."*
- *"For reproducible shuffling I pass the DataLoader its own torch.Generator, so data order
  doesn't depend on how many random numbers the model consumed."*

## 🔗 Next
We've got every piece: Module, loss, optimizer, loop, data. Next, M01E12: we put them
together on a real dataset, MNIST, hit 97%+ on the M4's GPU, and time CPU vs MPS honestly.

## 📹 Metadata
- **YouTube title:** `PyTorch DataLoader + collate_fn: batching text that doesn't line up | PyTorch from Zero #11`
- **Thumbnail (3–5 words):** `WORKERS MADE IT SLOWER`
- **Short (which beat, 45–60 s):** The measure table: num_workers 0 vs 2 vs 4, and why more workers was 3x slower.
- **X post:** "Set num_workers=4 on my DataLoader. It got 3x slower. Workers are processes; every batch gets pickled back. If __getitem__ is a dict lookup, the overhead is the whole cost. Start at 0. Measure. Then add workers."
- **Sources:** PyTorch: Learn the Basics, "Datasets & DataLoaders" (https://docs.pytorch.org/tutorials/beginner/basics/intro.html); `torch.utils.data` docs
