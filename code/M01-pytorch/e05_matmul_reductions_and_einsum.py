"""M01E05 · Matmul, reductions and einsum.

*  is elementwise.  @ is matmul (and batches over leading dims).
mm is 2-D only. bmm is 3-D only. einsum / einops write the shapes down for you.

Build: attention scores Q @ K^T for (B, H, T, d) three ways, all equal,
then finish attention and match F.scaled_dot_product_attention.
"""
import math
import time

import einops
import torch
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"


def scores_bmm(q, k):
    B, H, T, d = q.shape
    q3 = q.reshape(B * H, T, d)                       # bmm only takes 3-D
    k3 = k.reshape(B * H, T, d)
    s = torch.bmm(q3, k3.transpose(1, 2))             # (BH, T, d) @ (BH, d, T) -> (BH, T, T)
    return s.view(B, H, T, T)


def scores_einsum(q, k):
    # letters are dims. d appears on the left but not the right -> summed over
    return torch.einsum("bhqd,bhkd->bhqk", q, k)


def scores_einops(q, k):
    return einops.einsum(q, k, "b h q d, b h k d -> b h q k")


def attention(q, k, v):
    s = scores_einsum(q, k) / math.sqrt(q.shape[-1])  # (B, H, T, T)
    p = s.softmax(dim=-1)                             # each ROW sums to 1: dim=-1 is the keys
    return p @ v                                      # (B, H, T, T) @ (B, H, T, d) -> (B, H, T, d)


def reductions_tour():
    x = torch.arange(6.0).view(2, 3)                  # [[0,1,2],[3,4,5]]
    print("sum()        ", x.sum())
    print("sum(dim=0)   ", x.sum(dim=0))              # collapses dim 0 -> shape (3,)
    print("sum(dim=1)   ", x.sum(dim=1))              # collapses dim 1 -> shape (2,)
    print("keepdim      ", x.sum(dim=1, keepdim=True).shape)   # (2, 1): still broadcasts back
    print("x / rowsum   ", x / x.sum(dim=1, keepdim=True))
    print("max(dim=1)   ", x.max(dim=1))              # returns values AND indices
    a = torch.ones(2, 3)
    print("a * x shape  ", (a * x).shape, "  a @ x.T shape", (a @ x.T).shape)


def bench(fn, *args, iters=20):
    fn(*args)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    if device == "mps":
        torch.mps.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


if __name__ == "__main__":
    reductions_tour()
    B, H, T, d = 8, 12, 512, 64                       # GPT-2 small: 12 heads of 64
    q, k, v = (torch.randn(B, H, T, d, device=device) for _ in range(3))
    for fn in (scores_bmm, scores_einsum, scores_einops):
        print(f"{fn.__name__:<14} {bench(fn, q, k):6.2f} ms")
    print(f"{'q @ k.mT':<14} {bench(lambda a, b: a @ b.mT, q, k):6.2f} ms")
    print(f"{'my attention':<14} {bench(attention, q, k, v):6.2f} ms")
    print(f"{'sdpa':<14} {bench(F.scaled_dot_product_attention, q, k, v):6.2f} ms")


# ---------- the receipt ----------
def test_three_ways_agree():
    torch.manual_seed(0)
    q, k = torch.randn(2, 4, 16, 8, device=device), torch.randn(2, 4, 16, 8, device=device)
    ref = q @ k.transpose(-2, -1)
    for fn in (scores_bmm, scores_einsum, scores_einops):
        out = fn(q, k)
        assert out.shape == (2, 4, 16, 16)
        torch.testing.assert_close(out, ref)


def test_attention_matches_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, device=device) for _ in range(3))
    torch.testing.assert_close(attention(q, k, v), F.scaled_dot_product_attention(q, k, v))


def test_star_is_not_matmul():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert torch.equal(a * a, torch.tensor([[1.0, 4.0], [9.0, 16.0]]))
    assert torch.equal(a @ a, torch.tensor([[7.0, 10.0], [15.0, 22.0]]))
    assert torch.equal(torch.mm(a, a), a @ a)
