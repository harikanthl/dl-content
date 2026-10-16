"""M01E13 · Saving, loading and reproducibility

Kill training halfway, resume from a checkpoint, get the SAME loss curve, bit for bit.
Run:   .venv/bin/python code/M01-pytorch/e13_saving_loading_and_reproducibility.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e13_saving_loading_and_reproducibility.py -q
"""
import os
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed):
    random.seed(seed)  # python's random
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch CPU + all GPUs (MPS/CUDA) in one call


def make_data(n=2048, seed=123):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 20, generator=g)
    y = (X[:, :5].sum(1) > 0).long()  # simple rule the model can learn
    return TensorDataset(X, y)


def make_model():
    return nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))


# ---------------------------------------------------------------------------
# A checkpoint is everything that changes during training. Four things:
#   1. model weights      model.state_dict()
#   2. optimizer state    opt.state_dict()     (Adam's m and v, step counts)
#   3. where you are      epoch
#   4. randomness         global RNG (dropout) + the DataLoader's generator (shuffle order)
# ---------------------------------------------------------------------------
def save_checkpoint(path, model, opt, epoch, loader_gen):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,  # the NEXT epoch to run
        "torch_rng": torch.get_rng_state(),
        "loader_rng": loader_gen.get_state(),
    }, path)


def run(epochs, save_to=None, resume_from=None, stop_after=None,
        restore_optimizer=True, restore_rng=True, seed=0):
    """Returns the per-step training losses. stop_after=k simulates a crash after epoch k."""
    seed_everything(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(seed)  # owns the shuffle order
    loader = DataLoader(make_data(), batch_size=64, shuffle=True, generator=gen)
    start = 0

    if resume_from is not None:
        state = torch.load(resume_from)  # weights_only=True by default: tensors and plain containers only
        model.load_state_dict(state["model"])
        if restore_optimizer:
            opt.load_state_dict(state["optimizer"])
        if restore_rng:
            torch.set_rng_state(state["torch_rng"])
            gen.set_state(state["loader_rng"])
        start = state["epoch"]

    losses = []
    for epoch in range(start, epochs):
        model.train()
        for x, y in loader:
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if save_to is not None:
            save_checkpoint(save_to, model, opt, epoch + 1, gen)  # save at every epoch boundary
        if stop_after is not None and epoch + 1 == stop_after:
            break  # "the spot instance got preempted"
    return losses


def same_seed_twice(device, steps=50):
    """Determinism check: train twice with the same seed, return the max weight difference."""
    out = []
    for _ in range(2):
        seed_everything(0)
        m = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 10)).to(device)
        opt = torch.optim.Adam(m.parameters())
        x, y = torch.randn(512, 256).to(device), torch.randint(0, 10, (512,)).to(device)
        for _ in range(steps):
            opt.zero_grad()
            F.cross_entropy(m(x), y).backward()
            opt.step()
        out.append(torch.cat([p.detach().flatten().cpu() for p in m.parameters()]))
    return (out[0] - out[1]).abs().max().item()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        ck = os.path.join(d, "ckpt.pt")
        full = run(4)
        a = run(4, save_to=ck, stop_after=2)  # crash after epoch 2
        print(f"checkpoint size: {os.path.getsize(ck) / 1024:.1f} KB")
        b = run(4, resume_from=ck)
        c = run(4, resume_from=ck, restore_optimizer=False)
        r = run(4, resume_from=ck, restore_rng=False)
    k = len(a)
    print("steps per run:", len(full), "| crash after step", k)
    print("resumed == uninterrupted:", a + b == full)
    for name, cur in [("full resume", b), ("no optimizer state", c), ("no RNG state", r)]:
        diff = max(abs(p - q) for p, q in zip(cur, full[k:]))
        print(f"{name:20s} first loss {cur[0]:.5f} vs {full[k]:.5f} | max diff {diff:.2e}")
    print("same seed twice, CPU max |dw|:", same_seed_twice(torch.device("cpu")))
    if torch.backends.mps.is_available():
        print("same seed twice, MPS max |dw|:", same_seed_twice(torch.device("mps")))


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
def test_resume_reproduces_the_exact_curve(tmp_path):
    ck = tmp_path / "ckpt.pt"
    full = run(4)
    first = run(4, save_to=ck, stop_after=2)
    second = run(4, resume_from=ck)
    assert len(first) + len(second) == len(full)
    assert first + second == full  # exact float equality on CPU


def test_forgetting_optimizer_state_changes_the_curve(tmp_path):
    ck = tmp_path / "ckpt.pt"
    full = run(4)
    first = run(4, save_to=ck, stop_after=2)
    second = run(4, resume_from=ck, restore_optimizer=False)
    assert first + second != full


def test_forgetting_rng_state_changes_the_curve(tmp_path):
    ck = tmp_path / "ckpt.pt"
    full = run(4)
    first = run(4, save_to=ck, stop_after=2)
    second = run(4, resume_from=ck, restore_rng=False)
    assert first + second != full


def test_checkpoint_loads_with_weights_only(tmp_path):
    ck = tmp_path / "ckpt.pt"
    run(1, save_to=ck)
    state = torch.load(ck, weights_only=True)
    assert set(state) == {"model", "optimizer", "epoch", "torch_rng", "loader_rng"}
    assert state["epoch"] == 1 and "state" in state["optimizer"]
