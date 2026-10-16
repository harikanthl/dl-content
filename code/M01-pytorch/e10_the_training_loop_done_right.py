"""M01E10 · The training loop, done right

One reusable train() with a proper validation pass. Every later module reuses this shape.
Run:   .venv/bin/python code/M01-pytorch/e10_the_training_loop_done_right.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e10_the_training_loop_done_right.py -q
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Data: 3 interleaved spirals. Not linearly separable, so the MLP has to work.
# ---------------------------------------------------------------------------
def make_spirals(n_per_class=500, n_classes=3, noise=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.linspace(0.0, 1.0, n_per_class).repeat(n_classes)  # (N,)
    k = torch.arange(n_classes).repeat_interleave(n_per_class)  # (N,) class id
    theta = k * 2 * math.pi / n_classes + r * 4 + noise * torch.randn(len(r), generator=g)
    X = torch.stack([r * theta.sin(), r * theta.cos()], dim=1)  # (N, 2)
    return X, k


def batches(X, y, batch_size, shuffle, generator=None):
    """Poor man's DataLoader (the real one is E11)."""
    idx = torch.randperm(len(X), generator=generator) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), batch_size):
        j = idx[i : i + batch_size]
        yield X[j], y[j]


def make_model():
    # Dropout and BatchNorm behave differently in train vs eval. That's the point.
    return nn.Sequential(
        nn.Linear(2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 3),
    )


# ---------------------------------------------------------------------------
# The canonical loop
# ---------------------------------------------------------------------------
@torch.no_grad()  # no graph, no grad memory: eval is pure inference
def evaluate(model, loader, device):
    model.eval()  # dropout off, batchnorm uses running stats
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()  # sum, divide once at the end
        correct += (logits.argmax(-1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def train(model, make_train_loader, make_val_loader, opt, epochs, device, log_every=1):
    """make_*_loader: zero-arg callables returning a fresh iterable of (x, y) each epoch."""
    history = []
    for epoch in range(epochs):
        model.train()  # undo the eval() from last epoch's validation
        t0, run_loss, n = time.perf_counter(), 0.0, 0
        for x, y in make_train_loader():
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)  # .item(): a float, NOT the tensor (that keeps the graph)
            n += len(y)
        val_loss, val_acc = evaluate(model, make_val_loader(), device)
        rec = dict(epoch=epoch, train_loss=run_loss / n, val_loss=val_loss, val_acc=val_acc,
                   sec=time.perf_counter() - t0)
        history.append(rec)
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d} | train {rec['train_loss']:.4f} | val {val_loss:.4f} "
                  f"| acc {val_acc:.3f} | {rec['sec']:.2f}s")
    return history


def main(epochs=40, device=None, seed=0):
    torch.manual_seed(seed)
    device = device or get_device()
    X, y = make_spirals(seed=seed)
    perm = torch.randperm(len(X))
    tr, va = perm[:1200], perm[1200:]  # 80/20 split, shuffled first (data is sorted by class!)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    return train(
        model,
        lambda: batches(X[tr], y[tr], 64, shuffle=True, generator=g),
        lambda: batches(X[va], y[va], 256, shuffle=False),
        opt, epochs, device, log_every=5,
    ), model


if __name__ == "__main__":
    hist, _ = main()
    print(f"final val acc {hist[-1]['val_acc']:.3f}")


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
def test_training_reaches_high_val_accuracy():
    hist, _ = main(epochs=30, device=torch.device("cpu"))
    assert hist[-1]["val_acc"] > 0.95
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_eval_mode_is_deterministic_train_mode_is_not():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(32, 2)
    model.train()
    assert not torch.equal(model(x), model(x))  # dropout: different masks every call
    model.eval()
    assert torch.equal(model(x), model(x))


def test_evaluate_leaves_eval_mode_and_builds_no_graph():
    model = make_model()
    X, y = make_spirals(50)
    evaluate(model, batches(X, y, 32, shuffle=False), torch.device("cpu"))
    assert not model.training  # evaluate leaves eval mode on: train() must call model.train()
    with torch.no_grad():
        assert not model(X).requires_grad
