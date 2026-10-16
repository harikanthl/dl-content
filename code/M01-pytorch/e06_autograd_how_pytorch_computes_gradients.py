"""M01E06 · Autograd: how PyTorch computes gradients.

Every op on a requires_grad tensor records a node in a graph (grad_fn).
.backward() walks that graph in reverse, applying the chain rule,
and ADDS the result into each leaf's .grad.

Build: f(a, b) = (a*b + b)^2. Gradient three ways: autograd, by hand, finite differences.
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"


def f(a, b):
    c = a * b          # c = ab
    d = c + b          # d = ab + b
    return d ** 2      # f = (ab + b)^2


def grad_by_hand(a, b):
    # chain rule: df/dd = 2d,  dd/da = b,  dd/db = a + 1
    d = a * b + b
    return 2 * d * b, 2 * d * (a + 1)


def grad_finite_diff(fn, a, b, h=1e-6):
    # central difference: (f(x+h) - f(x-h)) / 2h, error ~ h^2. needs float64.
    da = (fn(a + h, b) - fn(a - h, b)) / (2 * h)
    db = (fn(a, b + h) - fn(a, b - h)) / (2 * h)
    return da, db


def grad_autograd(a_val, b_val, dev="cpu"):
    a = torch.tensor(a_val, requires_grad=True, device=dev)
    b = torch.tensor(b_val, requires_grad=True, device=dev)
    out = f(a, b)
    out.backward()
    return a.grad.item(), b.grad.item()


if __name__ == "__main__":
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    out = f(a, b)
    print("f =", out.item())                                  # (6 + 2)^2 = 64
    print("graph:", out.grad_fn, "->", out.grad_fn.next_functions)
    out.backward()
    print("autograd   :", a.grad.item(), b.grad.item())      # 32, 64
    print("by hand    :", grad_by_hand(3.0, 2.0))
    print("finite diff:", grad_finite_diff(f, 3.0, 2.0))

    # trap 1: grads ACCUMULATE
    f(a, b).backward()
    print("after 2nd backward, a.grad =", a.grad.item(), "(doubled, not reset)")
    a.grad = None; b.grad = None                              # what zero_grad() does

    # trap 2: backward twice through the same graph
    out = f(a, b)
    out.backward()
    try:
        out.backward()
    except RuntimeError as e:
        print("2nd backward:", str(e).split(".")[0])

    # no_grad and detach: stop recording
    with torch.no_grad():
        print("under no_grad, requires_grad =", f(a, b).requires_grad)
    print("detach requires_grad =", (a * b).detach().requires_grad)

    # trap 3: .grad only lives on leaves
    c = a * b
    print("is_leaf a, c:", a.is_leaf, c.is_leaf)
    print("on", device, ":", grad_autograd(3.0, 2.0, device))


# ---------- the receipt ----------
def test_autograd_matches_hand_and_finite_diff():
    for a_val, b_val in [(3.0, 2.0), (-1.5, 0.7), (0.2, -4.0)]:
        auto = grad_autograd(a_val, b_val)
        hand = grad_by_hand(a_val, b_val)
        fd = grad_finite_diff(f, a_val, b_val)                # python floats are float64
        torch.testing.assert_close(torch.tensor(auto), torch.tensor(hand))
        torch.testing.assert_close(torch.tensor(hand, dtype=torch.float64),
                                   torch.tensor(fd, dtype=torch.float64), rtol=1e-6, atol=1e-6)


def test_gradcheck():
    # PyTorch's own finite-difference checker, float64, on tensors of any shape
    a = torch.randn(5, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(f, (a, b))


def test_grads_accumulate():
    a = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(2.0)
    f(a, b).backward()
    f(a, b).backward()
    assert a.grad.item() == 64.0                              # 32 + 32


def test_works_on_device():
    torch.testing.assert_close(torch.tensor(grad_autograd(3.0, 2.0, device)), torch.tensor([32.0, 64.0]))
