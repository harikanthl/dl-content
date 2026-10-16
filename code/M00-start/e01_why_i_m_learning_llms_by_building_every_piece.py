"""M00E01 · Why I'm learning LLMs by building every piece.

Two things on camera:
  1. The map: read curriculum.json and print every module with its episode count.
  2. The receipt rule: every from-scratch piece gets checked against PyTorch.
     We write softmax twice. The naive one looks right and is wrong.
     The receipt catches it.

Run:   .venv/bin/python code/M00-start/e01_why_i_m_learning_llms_by_building_every_piece.py
Test:  .venv/bin/python -m pytest code/M00-start/e01_why_i_m_learning_llms_by_building_every_piece.py -q
"""
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
device = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------- 1. The map ----------
def print_map():
    cur = json.loads((REPO / "curriculum" / "curriculum.json").read_text())
    for m in cur["modules"]:
        print(f"{m['code']}  {m['episodes']:>3} eps  {m['name']:<45} {m['hardware']}")
    print(f"total: {len(cur['episodes'])} episodes")
    return cur


# ---------- 2. The receipt rule ----------
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
        return True
    except AssertionError as err:
        print(f"RECEIPT  {name:<16} FAIL  ({str(err).splitlines()[0]})")
        return False


if __name__ == "__main__":
    print_map()
    print(f"\ndevice: {device}\n")

    torch.manual_seed(0)
    small = torch.randn(4, 8, device=device)          # normal logits
    big = small * 100                                  # logits like a model that got confident

    ref_small, ref_big = torch.softmax(small, -1), torch.softmax(big, -1)

    receipt(softmax_naive(small), ref_small, "naive, small")   # passes. looks done.
    receipt(softmax_naive(big), ref_big, "naive, big")         # fails. exp(800) = inf -> nan
    receipt(softmax_stable(big), ref_big, "stable, big")       # passes.
    print("naive big has nan:", torch.isnan(softmax_naive(big)).any().item())


# ---------- the receipt, as a test ----------
def test_map_has_every_module():
    cur = print_map()
    assert len(cur["modules"]) == 18
    assert sum(m["episodes"] for m in cur["modules"]) == len(cur["episodes"]) == 216


def test_naive_softmax_breaks_on_big_logits():
    x = torch.tensor([[1000.0, 999.0, 0.0]], device=device)
    assert torch.isnan(softmax_naive(x)).any()          # the bug is real


def test_stable_softmax_matches_pytorch():
    torch.manual_seed(0)
    for scale in (1.0, 100.0, 1e4):
        x = torch.randn(32, 128, device=device) * scale
        torch.testing.assert_close(softmax_stable(x), torch.softmax(x, -1))
