"""M01E09 · Loss functions and optimizers

Cross-entropy by hand, then SGD-with-momentum by hand, matched to torch step for step.
Run:   .venv/bin/python code/M01-pytorch/e09_loss_functions_and_optimizers.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e09_loss_functions_and_optimizers.py -q
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Losses. A loss turns "how wrong" into one scalar we can call .backward() on.
# ---------------------------------------------------------------------------
def mse(pred, target):  # regression: (B,) vs (B,)
    return ((pred - target) ** 2).mean()


def cross_entropy(logits, target):  # classification: logits (B, C), target (B,) int64
    # -log softmax(logits)[target] = logsumexp(logits) - logits[target]
    # logsumexp subtracts the max internally, so big logits don't overflow exp().
    lse = torch.logsumexp(logits, dim=-1)  # (B,)
    correct = logits.gather(1, target[:, None]).squeeze(1)  # (B,) pick the right class
    return (lse - correct).mean()


# ---------------------------------------------------------------------------
# 2. SGD with momentum, exactly the way torch.optim.SGD does it.
#    g  = grad + wd * p               (weight decay folded into the gradient)
#    b  = g                on step 1  (no warm-up from zero)
#    b  = mu * b + g       after that  (velocity: running sum of past grads)
#    p  = p - lr * b
# ---------------------------------------------------------------------------
class MySGD:
    def __init__(self, params, lr, momentum=0.0, weight_decay=0.0):
        self.params = list(params)  # materialize: a generator would be empty on step 2
        self.lr, self.mu, self.wd = lr, momentum, weight_decay
        self.buf = [None] * len(self.params)  # one velocity tensor per parameter

    @torch.no_grad()  # the update itself must not be recorded by autograd
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.add(p, alpha=self.wd) if self.wd else p.grad  # g + wd*p
            if self.mu:
                if self.buf[i] is None:
                    self.buf[i] = g.clone()  # clone: don't alias p.grad
                else:
                    self.buf[i].mul_(self.mu).add_(g)
                g = self.buf[i]
            p.add_(g, alpha=-self.lr)  # p -= lr*g, in place: the Parameter object must stay the same

    def zero_grad(self):
        for p in self.params:
            p.grad = None  # torch's default (set_to_none=True): cheaper than filling zeros


# ---------------------------------------------------------------------------
# 3. The order that matters: zero_grad -> forward -> loss -> backward -> step
# ---------------------------------------------------------------------------
def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))


def run(opt_name, steps=20):
    torch.manual_seed(1)
    X, y = torch.randn(64, 10), torch.randint(0, 3, (64,))
    model = make_model()
    kw = dict(lr=0.1, momentum=0.9, weight_decay=1e-4)
    opt = MySGD(model.parameters(), **kw) if opt_name == "mine" else torch.optim.SGD(model.parameters(), **kw)
    losses = []
    for _ in range(steps):
        opt.zero_grad()  # 1. clear last step's grads (they ACCUMULATE otherwise)
        loss = cross_entropy(model(X), y)  # 2-3. forward + loss
        loss.backward()  # 4. fill p.grad for every parameter
        opt.step()  # 5. move the weights
        losses.append(loss.item())
    return losses, model


if __name__ == "__main__":
    logits = torch.tensor([[2.0, 1.0, 0.1]])
    print("CE mine:", cross_entropy(logits, torch.tensor([0])).item(),
          "torch:", F.cross_entropy(logits, torch.tensor([0])).item())
    # The classic bug: softmax first, then CrossEntropyLoss (which softmaxes again).
    print("CE on softmax(logits) (wrong):", F.cross_entropy(logits.softmax(-1), torch.tensor([0])).item())
    print("Big logits, naive log(softmax):", torch.log(torch.softmax(torch.tensor([[1000.0, 0.0]]), -1)))
    mine, _ = run("mine")
    ref, _ = run("torch")
    for i in (0, 1, 5, 19):
        print(f"step {i:2d}  mine {mine[i]:.6f}  torch {ref[i]:.6f}")


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits, target = torch.randn(32, 10) * 5, torch.randint(0, 10, (32,))
    torch.testing.assert_close(cross_entropy(logits, target), F.cross_entropy(logits, target))
    big = torch.tensor([[1000.0, 0.0]])  # would overflow a naive exp
    assert torch.isfinite(cross_entropy(big, torch.tensor([1])))


def test_mse_matches_torch():
    a, b = torch.randn(50), torch.randn(50)
    torch.testing.assert_close(mse(a, b), F.mse_loss(a, b))


def test_my_sgd_matches_torch_step_for_step():
    mine, m1 = run("mine", steps=50)
    ref, m2 = run("torch", steps=50)
    assert mine == ref  # bit-identical, not just close
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
    assert mine[-1] < mine[0] * 0.5  # and it actually learns


def test_grads_accumulate_without_zero_grad():
    model = make_model()
    x = torch.randn(8, 10)
    model(x).sum().backward()
    g1 = model[0].weight.grad.clone()
    model(x).sum().backward()  # no zero_grad in between
    torch.testing.assert_close(model[0].weight.grad, 2 * g1)
