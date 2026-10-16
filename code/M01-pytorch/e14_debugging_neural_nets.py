"""M01E14 · Debugging neural nets

Four broken models, four tools. Each bug is silent (or nearly). Each tool catches one.
  Bug 1: shape broadcast in the loss      -> Tool: shape asserts
  Bug 2: softmax before cross_entropy     -> Tool: overfit one batch
  Bug 3: forgot zero_grad                 -> Tool: gradient norms
  Bug 4: naive log(softmax) makes NaN     -> Tool: forward hooks that check isfinite
Run:   .venv/bin/python code/M01-pytorch/e14_debugging_neural_nets.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e14_debugging_neural_nets.py -q
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(n_in=20, n_out=10, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, n_out))


# ---------------------------------------------------------------------------
# Bug 1: pred is (B, 1), target is (B,). (B,1) - (B,) broadcasts to (B, B).
# The loss is a real number, it goes down, and the model learns the wrong thing.
# ---------------------------------------------------------------------------
def regression_loss(pred, target, check=True):
    if check:  # Tool 1: assert shapes at boundaries. One line, catches a week of pain.
        assert pred.shape == target.shape, f"pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
    return ((pred - target) ** 2).mean()


def train_regression(check, steps=300):
    torch.manual_seed(0)
    X = torch.randn(256, 20)
    y = X @ torch.randn(20)  # (B,) a perfectly learnable linear target
    model = mlp(n_out=1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = regression_loss(model(X), y, check=check)  # model(X) is (B, 1)!
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return F.mse_loss(model(X).squeeze(1), y).item()  # the TRUE error


# ---------------------------------------------------------------------------
# Tool 2: overfit one batch. A healthy model drives loss to ~0 on 32 examples.
# If it can't memorize 32 examples, nothing else matters yet.
# ---------------------------------------------------------------------------
def overfit_one_batch(model, x, y, loss_fn, steps=500, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def ce_correct(logits, y):
    return F.cross_entropy(logits, y)


def ce_double_softmax(logits, y):  # Bug 2: cross_entropy softmaxes AGAIN
    return F.cross_entropy(logits.softmax(-1), y)


# ---------------------------------------------------------------------------
# Tool 3: gradient norms. Bug 3: no zero_grad, so grads pile up step after step.
# ---------------------------------------------------------------------------
def grad_norm(model):
    return torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)).item()


def train_sgd(zero_grad, steps=20):
    torch.manual_seed(0)
    x, y = torch.randn(64, 20), torch.randint(0, 10, (64,))
    model = mlp()
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    norms, losses = [], []
    for _ in range(steps):
        if zero_grad:
            opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        norms.append(grad_norm(model))  # log this every step in real training too
        losses.append(loss.item())
        opt.step()
    return norms, losses


# ---------------------------------------------------------------------------
# Tool 4: forward hooks that report the FIRST module whose output is not finite.
# Bug 4: log(softmax(x)) underflows to log(0) = -inf for confident logits.
# ---------------------------------------------------------------------------
class NaiveLogSoftmax(nn.Module):
    def forward(self, x):
        return torch.log(torch.softmax(x, dim=-1))  # BUG: use F.log_softmax


def add_nan_hooks(model):
    bad, handles = [], []
    for name, m in model.named_modules():
        def hook(mod, inp, out, name=name):  # name=name: freeze the loop variable
            if not torch.isfinite(out).all() and not bad:
                bad.append(name)
        handles.append(m.register_forward_hook(hook))
    return bad, handles


def find_nan():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), NaiveLogSoftmax())
    with torch.no_grad():
        model[0].weight.mul_(1000)  # big weights -> confident logits, like a model mid-blowup
    bad, handles = add_nan_hooks(model)
    out = model(torch.randn(8, 4))
    for h in handles:
        h.remove()  # hooks live forever unless removed
    return (bad[0] if bad else None), out


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"Bug 1 | true MSE without assert: {train_regression(check=False):.3f}")
    try:
        train_regression(check=True)
    except AssertionError as e:
        print("Bug 1 | assert caught it:", e)

    torch.manual_seed(1)
    x, y = torch.randn(32, 20), torch.randint(0, 10, (32,))
    print(f"Bug 2 | overfit one batch, correct CE: {overfit_one_batch(mlp(), x, y, ce_correct):.4f}")
    print(f"Bug 2 | overfit one batch, double softmax: {overfit_one_batch(mlp(), x, y, ce_double_softmax):.4f}"
          f"  (floor = log(1 + 9/e) = {torch.log(torch.tensor(1 + 9 / torch.e)):.4f})")

    good, bad_ = train_sgd(True), train_sgd(False)
    print("Bug 3 | grad norm, step 1/10/20 with zero_grad:", [round(good[0][i], 2) for i in (0, 9, 19)])
    print("Bug 3 | grad norm, step 1/10/20 without     :", [round(bad_[0][i], 2) for i in (0, 9, 19)])
    print("Bug 3 | loss at step 20 with / without:", round(good[1][-1], 3), round(bad_[1][-1], 3))

    culprit, out = find_nan()
    print("Bug 4 | first non-finite output from module:", culprit, "| sample row:", out[0].tolist())


# ---------------------------------------------------------------------------
# Receipts: every bug is caught by its tool, and every fix passes.
# ---------------------------------------------------------------------------
def test_bug1_broadcast_learns_wrong_thing_and_assert_catches_it():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert train_regression(check=False) > 1.0  # trained 300 steps, still bad
    try:
        train_regression(check=True)
        raise RuntimeError("assert should have fired")
    except AssertionError as e:
        assert "(256, 1)" in str(e) and "(256,)" in str(e)


def test_bug1_fixed():
    torch.manual_seed(0)
    X = torch.randn(256, 20); y = X @ torch.randn(20)
    model = mlp(n_out=1); opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(300):
        loss = regression_loss(model(X).squeeze(1), y)  # the fix: squeeze(1)
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1


def test_bug2_double_softmax_cannot_overfit():
    torch.manual_seed(1)
    x, y = torch.randn(32, 20), torch.randint(0, 10, (32,))
    assert overfit_one_batch(mlp(), x, y, ce_correct) < 0.01
    floor = torch.log(torch.tensor(1 + 9 / torch.e)).item()  # ~1.4612 for 10 classes
    assert overfit_one_batch(mlp(), x, y, ce_double_softmax) > floor - 1e-3


def test_bug3_grad_norm_explodes_without_zero_grad():
    good, bad = train_sgd(True), train_sgd(False)
    assert bad[0][-1] > 5 * good[0][-1]


def test_bug4_hook_finds_the_naive_log_softmax():
    culprit, out = find_nan()
    assert culprit == "2"  # index of NaiveLogSoftmax in the Sequential
    assert not torch.isfinite(out).all()
    x = torch.tensor([[1000.0, 0.0]])
    assert torch.isfinite(F.log_softmax(x, -1)).all()  # the fix
