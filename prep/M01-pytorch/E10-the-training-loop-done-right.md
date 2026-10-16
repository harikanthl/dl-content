# EP012 · M01E10 · The training loop, done right

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e10_the_training_loop_done_right.py` · **Explainer:** `explainers/scenes/m01e10_the_training_loop_done_right.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "My validation loss is lower than my training loss. That's not a bug. If you know why,
> you understand the training loop. If you don't, this is the video."

## ❓ The question (why this matters, 45 s)
Every training script you'll ever read, nanoGPT, Hugging Face's Trainer, the Llama
recipes, is the same loop: train for a while, stop, evaluate on held-out data, log,
repeat. The loop has maybe 25 lines. About five of them are easy to get wrong, and none of
those five raise an error. You get a model that's slightly worse, or a validation number
that's slightly lying, or a memory leak that kills a run at hour six.

We write it once, correctly, as a `train()` function. M03 through M06 reuse this exact
shape. The nanoGPT loop in M06 is this plus gradient accumulation and a scheduler.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: practice vs the exam.**
> Training is practice. During practice the coach makes it harder on purpose: dropout
> randomly benches 20% of your neurons every rep, so no single neuron gets lazy.
> Evaluation is the exam. On exam day everyone plays, nobody's benched, and you don't
> take notes on your mistakes (no gradients). `model.train()` and `model.eval()` are the
> coach blowing the whistle to switch between the two.
>
> That's the hook answered: training loss is measured *during practice, with handicaps*.
> Validation loss is measured *at the exam, at full strength*. So val can be lower.

The picture: two boxes side by side. Left box "train": dropout masks flickering, BatchNorm
using batch stats, gradient arrows flowing back. Right box "eval": all neurons lit,
BatchNorm using frozen running stats, no arrows.

## 🪤 The trap
**`model.eval()` and `torch.no_grad()` are different switches. You need both.**

| | what it changes | forget it and... |
|---|---|---|
| `model.eval()` | Dropout off, BatchNorm uses running stats | eval numbers are noisy and pessimistic; BatchNorm stats get updated *by your validation data* |
| `torch.no_grad()` | autograd stops recording | same numbers, but every val batch builds a graph: slower, way more memory |

And the sneaky third one: **`evaluate()` calls `model.eval()`, so the next epoch must call
`model.train()` again.** Put `model.train()` at the top of the epoch, not once before the
loop. If it's outside the loop, epoch 1 trains with dropout and epochs 2+ train
without it. No error. Just a worse model.

## 📐 The math (derive on screen, 2–3 min)
Not much math today. Two pieces of bookkeeping that people get wrong:

1. **Epoch loss is a weighted mean.** With 1200 examples and batch 64 you get 18 full
   batches and one batch of 48. Averaging the 19 batch means gives the last batch the same
   weight as a full one. Correct:
   `epoch_loss = Σ_b (loss_b · n_b) / Σ_b n_b`.
   In eval we go one better: `reduction="sum"` per batch, divide by N once at the end.
2. **Dropout's scaling.** In train mode, each activation is zeroed with prob `p` and the
   survivors are multiplied by `1/(1−p)`. So `E[output] = input` in both modes, which is
   why eval can just turn it off. (This is "inverted dropout". Mention the name.)

Shapes: `x (B, 2)`, logits `(B, 3)`, `y (B,)`. Accuracy = `mean(argmax(logits, -1) == y)`.

## 💻 Code it (from scratch, 4–5 min)
```python
@torch.no_grad()                                   # switch 2: no graph
def evaluate(model, loader, device):
    model.eval()                                   # switch 1: dropout off, BN frozen
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n

def train(model, make_train_loader, make_val_loader, opt, epochs, device):
    history = []
    for epoch in range(epochs):
        model.train()                              # INSIDE the loop: evaluate() turned it off
        run_loss, n = 0.0, 0
        for x, y in make_train_loader():
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)       # .item(): a float, frees the graph
            n += len(y)
        val_loss, val_acc = evaluate(model, make_val_loader(), device)
        history.append(dict(epoch=epoch, train_loss=run_loss / n, val_loss=val_loss, val_acc=val_acc))
    return history
```
Data: three interleaved spirals (1,500 points, 2D), split 1,200 / 300 after a shuffle.
Model: `2 → 128 → 128 → 3` with BatchNorm and Dropout(0.2), so both mode switches matter.
AdamW, lr 3e-3. The `batches()` helper is a 6-line stand-in for DataLoader (next episode).

Why `make_train_loader` is a callable: a generator is used up after one pass. Passing a
function that builds a fresh one each epoch is the honest interface. A real `DataLoader`
is re-iterable, so you can also pass `lambda: loader`.

## ✅ The receipt (1 min)
- `test_training_reaches_high_val_accuracy`: 30 epochs on CPU, val accuracy > 0.95.
- `test_eval_mode_is_deterministic_train_mode_is_not`: same input twice in train mode
  gives different outputs (dropout masks), in eval mode gives identical outputs.
- `test_evaluate_leaves_eval_mode_and_builds_no_graph`: after `evaluate()`,
  `model.training` is False. This is the test that justifies `model.train()` inside the loop.

`pytest -q` → **3 passed** in about 1.5 s.

## 📊 Measure (1 min)
Measured on M4 Mac mini:
- Val accuracy after 40 epochs: **0.98** on CPU (≈0.98–1.00 on MPS; 300 val points, so
  one point is 0.3%).
- Train loss sits *above* val loss the whole run (epoch 30: train 0.12, val 0.07). That's
  dropout. The hook, receipted.
- Same trained model scored on all 1,500 points: **0.980 in train mode, 0.993 in eval
  mode**. Forget `model.eval()` and you under-report your own model.
- Time per epoch: **CPU 0.010 s, MPS 0.035 s**. MPS's first epoch: **3.3 s** (one-time
  kernel compile and device warm-up). For an 18k-parameter model, the GPU loses. Kernel
  launch overhead beats the math. We'll see the crossover in E12 with a real model.

## ⚠️ Gotchas
- **`run_loss += loss` (no `.item()`).** Keeps every step's graph alive through the sum.
  Memory climbs every step. On a big model this is the "OOM at hour six" bug.
- **`model.train()` outside the epoch loop.** Silent. Only epoch 1 uses dropout.
- **Evaluating without `no_grad`.** Same numbers, but every activation is kept for a backward that never comes. On an LLM val pass,
  that's the difference between fitting and OOM.
- **Not shuffling before the split.** `make_spirals` returns data sorted by class. Take
  the first 80% unshuffled and class 2 is barely in train. Val accuracy tanks, nothing errors.
- **Mean of batch means.** Off by a little when the last batch is short. Weight by batch size.
- **Timing the first epoch on MPS.** It includes warm-up. Report steady-state.
- `torch.inference_mode()` is a stricter, slightly faster `no_grad()`. Fine for eval; use
  `no_grad` if you might need the outputs in a later autograd computation.

## 🎤 Interview lines
- *"model.eval() changes layer behavior, dropout and batchnorm. no_grad() changes
  autograd, it stops building the graph. Evaluation needs both, for different reasons."*
- *"Training loss is measured with dropout on, so it can be higher than validation loss.
  That's expected, not leakage."*
- *"I accumulate loss.item(), not the loss tensor, so I'm not holding the graph across
  steps, and I weight by batch size so the short last batch doesn't skew the mean."*
- *"I call model.train() at the start of every epoch because my eval function puts the
  model in eval mode."*

## 🔗 Next
Our `batches()` helper is six lines of index slicing. It won't survive variable-length text.
Next, M01E11: `Dataset` and `DataLoader`, and a `collate_fn` that pads a batch of sentences.

## 📹 Metadata
- **YouTube title:** `The PyTorch training loop, done right (eval vs no_grad, explained) | PyTorch from Zero #10`
- **Thumbnail (3–5 words):** `VAL < TRAIN?!`
- **Short (which beat, 45–60 s):** The hook + ELI5: val loss below train loss, practice vs exam, dropout benches players.
- **X post:** "My val loss is lower than my train loss and that's correct. Train loss is measured with dropout on. Also: model.eval() and torch.no_grad() are different switches. You need both. Same model, same data: 98.0% acc in train mode, 99.3% in eval mode."
- **Sources:** PyTorch: Learn the Basics, "Optimizing Model Parameters" (https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
