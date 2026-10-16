"""M01E07 · Linear regression with raw autograd.

No nn. No optim. Two tensors, a loss, .backward(), and a manual update.
Recover w=3, b=2 from noisy data. This loop is every training loop you'll
ever write, with the boilerplate peeled off.
"""
from pathlib import Path

import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"


def make_data(n=200, w_true=3.0, b_true=2.0, noise=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g) * 4 - 2                    # x in [-2, 2], shape (n,)
    y = w_true * x + b_true + noise * torch.randn(n, generator=g)
    return x.to(device), y.to(device)


def train(x, y, lr=0.1, steps=200):
    w = torch.zeros((), device=device, requires_grad=True)   # scalar leaf tensors
    b = torch.zeros((), device=device, requires_grad=True)
    losses = []
    for _ in range(steps):
        pred = w * x + b                                      # forward, shape (n,)
        loss = ((pred - y) ** 2).mean()                       # MSE
        loss.backward()                                       # fills w.grad, b.grad
        with torch.no_grad():                                 # the update is not part of the model
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad = None                                         # grads accumulate: reset them
        b.grad = None
        losses.append(loss.item())
    return w.detach(), b.detach(), losses


def closed_form(x, y):
    # least squares answer, no iteration: solve [x 1] @ [w b] = y
    A = torch.stack([x, torch.ones_like(x)], dim=1).cpu()
    sol = torch.linalg.lstsq(A, y.cpu()[:, None]).solution
    return sol[0, 0], sol[1, 0]


def plot(losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 3.5))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE (log)")
    plt.title("Linear regression, raw autograd")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    x, y = make_data()
    w, b, losses = train(x, y)
    w_cf, b_cf = closed_form(x, y)
    for step in (0, 1, 5, 10, 25, 50, 100, 199):
        print(f"step {step:>3}  loss {losses[step]:.4f}")
    print(f"gradient descent: w={w.item():.4f} b={b.item():.4f}")
    print(f"closed form     : w={w_cf.item():.4f} b={b_cf.item():.4f}")
    true_loss = ((3.0 * x + 2.0 - y) ** 2).mean().item()      # loss of the TRUE line
    print(f"final loss {losses[-1]:.4f}  vs  loss of true w=3,b=2: {true_loss:.4f}")
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    plot(losses, out / "e07_loss.png")


# ---------- the receipt ----------
def test_recovers_true_parameters():
    x, y = make_data()
    w, b, losses = train(x, y)
    assert abs(w.item() - 3.0) < 0.1 and abs(b.item() - 2.0) < 0.1
    assert losses[-1] < losses[0] / 10


def test_matches_closed_form_least_squares():
    x, y = make_data()
    w, b, _ = train(x, y, steps=500)
    w_cf, b_cf = closed_form(x, y)
    torch.testing.assert_close(w.cpu(), w_cf, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(b.cpu(), b_cf, rtol=1e-4, atol=1e-4)


def test_forgetting_zero_grad_breaks_it():
    # same loop, but the grads pile up: it diverges
    x, y = make_data()
    w = torch.zeros((), device=device, requires_grad=True)
    for _ in range(50):
        ((w * x - y) ** 2).mean().backward()
        with torch.no_grad():
            w -= 0.1 * w.grad
    assert not abs(w.item() - 3.0) < 0.1
