"""M01E04 · Broadcasting: the bug factory.

Rule 1: line shapes up from the RIGHT. Missing dims on the left count as 1.
Rule 2: each pair of dims must be equal, or one of them must be 1 (it gets stretched).

Build: pairwise distances with broadcasting, then reproduce the (N,) vs (N,1) bug.
"""
import time

import torch
import torch.nn.functional as F

device = "mps" if torch.backends.mps.is_available() else "cpu"


def broadcast_shape(a, b):
    # the two rules, by hand
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        x = a[-i] if i <= len(a) else 1               # rule 1: pad on the left with 1
        y = b[-i] if i <= len(b) else 1
        if x != y and 1 not in (x, y):                # rule 2
            raise ValueError(f"cannot broadcast {a} and {b} at dim -{i}: {x} vs {y}")
        out.append(max(x, y))
    return tuple(reversed(out))


def pairwise_dist(X, Y):
    # (N, 1, D) - (1, M, D) -> (N, M, D) -> sum over D -> (N, M)
    diff = X[:, None, :] - Y[None, :, :]
    return diff.pow(2).sum(-1).sqrt()


def pairwise_dist_expanded(X, Y):
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y   never builds the (N, M, D) tensor
    sq = (X * X).sum(-1)[:, None] + (Y * Y).sum(-1)[None, :] - 2 * X @ Y.T
    return sq.clamp_min(0).sqrt()                     # clamp: rounding can go slightly negative


def the_bug(n=5):
    torch.manual_seed(0)
    y = torch.randn(n)                  # targets, shape (N,)
    pred = y[:, None] + 0.1             # model output, shape (N, 1). off by 0.1 everywhere
    wrong = ((pred - y) ** 2).mean()    # (N,1) - (N,) -> (N, N) !!
    right = ((pred.squeeze(-1) - y) ** 2).mean()
    return (pred - y).shape, wrong.item(), right.item()


def bench(fn, *args, iters=10):
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
    print(broadcast_shape((8, 1, 6, 1), (7, 1, 5)))   # (8, 7, 6, 5)
    shape, wrong, right = the_bug()
    print(f"(pred - y).shape = {tuple(shape)}   wrong MSE = {wrong:.4f}   right MSE = {right:.4f}")
    F.mse_loss(torch.zeros(5, 1), torch.zeros(5))     # PyTorch warns you. read your warnings.

    X = torch.randn(1024, 128, device=device)
    Y = torch.randn(1024, 128, device=device)
    print(f"broadcast  {bench(pairwise_dist, X, Y):7.2f} ms  (builds 1024x1024x128 = 512 MB)")
    print(f"expanded   {bench(pairwise_dist_expanded, X, Y):7.2f} ms")
    print(f"torch.cdist{bench(torch.cdist, X, Y):7.2f} ms")


# ---------- the receipt ----------
def test_broadcast_shape_matches_torch():
    cases = [((3,), (3,)), ((5, 1), (5,)), ((8, 1, 6, 1), (7, 1, 5)), ((1,), (4, 4)), ((), (2, 3))]
    for a, b in cases:
        assert broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))
    for a, b in [((3,), (4,)), ((2, 3), (3, 2))]:
        try:
            broadcast_shape(a, b)
            raise AssertionError("should not broadcast")
        except ValueError:
            pass


def test_pairwise_matches_cdist():
    torch.manual_seed(0)
    X, Y = torch.randn(64, 16, device=device), torch.randn(48, 16, device=device)
    ref = torch.cdist(X, Y)
    torch.testing.assert_close(pairwise_dist(X, Y), ref)
    torch.testing.assert_close(pairwise_dist_expanded(X, Y), ref, atol=1e-3, rtol=1e-3)


def test_the_bug_is_silent_and_wrong():
    shape, wrong, right = the_bug()
    assert tuple(shape) == (5, 5)                     # no error raised. that's the problem.
    assert abs(right - 0.01) < 1e-6                   # 0.1 ** 2
    assert wrong > 10 * right
