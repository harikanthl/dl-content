"""M01E11 · Dataset and DataLoader

A Dataset over a text file (one example per line) and a padding collate_fn.
This is the exact shape of every LLM data pipeline later in the series, just character-level.
Run:   .venv/bin/python code/M01-pytorch/e11_dataset_and_dataloader.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e11_dataset_and_dataloader.py -q
"""
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

PAD = 0  # id 0 is reserved for padding, real characters start at 1


# ---------------------------------------------------------------------------
# 1. Dataset: the contract is two methods. __len__ and __getitem__. That's it.
# ---------------------------------------------------------------------------
class LineDataset(Dataset):
    def __init__(self, path):
        self.lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
        chars = sorted(set("".join(self.lines)))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # +1: keep 0 for PAD
        self.itos = {i: c for c, i in self.stoi.items()}

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, i):  # one example, any length
        return torch.tensor([self.stoi[c] for c in self.lines[i]], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids if int(i) != PAD)


# ---------------------------------------------------------------------------
# 2. collate_fn: turns a LIST of examples into ONE batch tensor.
#    The default collate calls torch.stack, which needs equal shapes. Ours pads.
# ---------------------------------------------------------------------------
def pad_collate(batch):
    lengths = torch.tensor([len(x) for x in batch])  # (B,)
    T = int(lengths.max())
    out = torch.full((len(batch), T), PAD, dtype=torch.long)  # (B, T) all PAD
    for b, x in enumerate(batch):
        out[b, : len(x)] = x  # right-pad: real tokens first
    mask = torch.arange(T)[None, :] < lengths[:, None]  # (1,T) < (B,1) -> (B,T) True on real tokens
    return out, lengths, mask


SAMPLE = """the cat sat
on the mat
a much longer line about dogs and cats
hi
pytorch pads batches
short one
transformers eat padded batches too
ok
"""


def write_sample(path):
    Path(path).write_text(SAMPLE)
    return path


def time_loader(ds, num_workers, batch_size=64, epochs=1):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate,
                    num_workers=num_workers, persistent_workers=num_workers > 0)
    t0 = time.perf_counter()
    for _ in range(epochs):
        for _ in dl:
            pass
    return time.perf_counter() - t0


if __name__ == "__main__":  # REQUIRED on macOS with num_workers>0: workers re-import this file
    data_dir = Path(__file__).resolve().parents[2] / "data"  # DL-content/data (gitignored)
    data_dir.mkdir(exist_ok=True)
    path = write_sample(data_dir / "lines.txt")
    ds = LineDataset(path)
    print(len(ds), "lines, vocab", len(ds.stoi) + 1, "| ds[0] =", ds[0].tolist())

    g = torch.Generator().manual_seed(0)  # own generator: shuffling reproducible, independent of model RNG
    dl = DataLoader(ds, batch_size=3, shuffle=True, collate_fn=pad_collate, generator=g)
    for x, lengths, mask in dl:
        print(tuple(x.shape), lengths.tolist(), [ds.decode(r) for r in x])

    # Workers: separate processes that call __getitem__ in parallel.
    big = LineDataset(path)
    big.lines = big.lines * 5000  # 40k lines
    for w in (0, 2, 4):
        print(f"num_workers={w}: {time_loader(big, w, epochs=3):.2f}s for 3 epochs")


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
def test_len_and_getitem(tmp_path):
    ds = LineDataset(write_sample(tmp_path / "x.txt"))
    assert len(ds) == 8
    assert ds.decode(ds[0]) == "the cat sat"
    assert PAD not in ds[0].tolist()


def test_pad_collate_shapes_and_mask(tmp_path):
    ds = LineDataset(write_sample(tmp_path / "x.txt"))
    x, lengths, mask = pad_collate([ds[3], ds[0], ds[2]])  # "hi", "the cat sat", long one
    assert x.shape == (3, len(SAMPLE.splitlines()[2]))
    assert lengths.tolist() == [2, 11, 38]
    assert mask.sum().item() == 2 + 11 + 38
    assert (x[~mask] == PAD).all() and (x[mask] != PAD).all()
    assert ds.decode(x[0]) == "hi"


def test_default_collate_fails_on_ragged(tmp_path):
    ds = LineDataset(write_sample(tmp_path / "x.txt"))
    dl = DataLoader(ds, batch_size=4)  # no collate_fn
    try:
        next(iter(dl))
        raise AssertionError("expected a stack error")
    except RuntimeError as e:
        assert "stack expects each tensor to be equal size" in str(e)


def test_shuffle_is_reproducible_with_generator(tmp_path):
    ds = LineDataset(write_sample(tmp_path / "x.txt"))

    def order(seed):
        dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=pad_collate,
                        generator=torch.Generator().manual_seed(seed))
        return [x.tolist() for x, _, _ in dl]

    assert order(0) == order(0)
    assert order(0) != order(1)
    # every line appears exactly once per epoch
    seen = sorted(ds.decode(r) for x, _, _ in DataLoader(ds, batch_size=3, shuffle=True,
                  collate_fn=pad_collate) for r in x)
    assert seen == sorted(ds.lines)
