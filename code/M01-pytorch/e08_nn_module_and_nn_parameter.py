"""M01E08 · nn.Module and nn.Parameter

Build MyLinear from scratch and prove it is nn.Linear, bit for bit.
Run:   .venv/bin/python code/M01-pytorch/e08_nn_module_and_nn_parameter.py
Test:  .venv/bin/python -m pytest code/M01-pytorch/e08_nn_module_and_nn_parameter.py -q
"""
import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. The layer. A Module is a class with two jobs: hold state, define forward().
# ---------------------------------------------------------------------------
class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()  # forget this and the first self.x = ... assignment crashes
        # nn.Parameter = a tensor that says "I am learnable, track me".
        # Assigning it as an attribute is what registers it. That's the whole trick.
        # Shape is (out, in), same as nn.Linear, so y = x @ W.T
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Same init as nn.Linear: Kaiming-uniform with a=sqrt(5),
        # which works out to U(-1/sqrt(fan_in), 1/sqrt(fan_in)).
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[1])  # fan_in
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):  # x: (..., in) -> (..., out)
        y = x @ self.weight.T  # (..., in) @ (in, out) -> (..., out)
        if self.bias is not None:
            y = y + self.bias  # (out,) broadcasts over the leading dims
        return y

    def extra_repr(self):  # what print(model) shows inside the parentheses
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}"


# ---------------------------------------------------------------------------
# 2. The trap: a plain tensor attribute is invisible to Module.
# ---------------------------------------------------------------------------
class BrokenLinear(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.weight = torch.randn(n_out, n_in, requires_grad=True)  # NOT a Parameter
        self.layers = [nn.Linear(n_in, n_in)]  # plain list: also invisible

    def forward(self, x):
        return self.layers[0](x) @ self.weight.T


# ---------------------------------------------------------------------------
# 3. Buffers: state that is saved and moved with .to(), but not trained.
# ---------------------------------------------------------------------------
class Scaler(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n))  # in state_dict, not in parameters()

    def forward(self, x):
        if self.training:  # model.train() / model.eval() flip this flag on every submodule
            self.running_mean.mul_(0.9).add_(0.1 * x.mean(0).detach())
        return x - self.running_mean


def tour():
    torch.manual_seed(0)
    lin = MyLinear(4, 3)
    print("parameters:", [(n, tuple(p.shape)) for n, p in lin.named_parameters()])
    print("state_dict keys:", list(lin.state_dict().keys()))
    print("param count:", sum(p.numel() for p in lin.parameters()))  # 4*3 + 3 = 15

    bad = BrokenLinear(4, 3)
    print("BrokenLinear parameters:", list(bad.named_parameters()))  # [] -> optimizer sees nothing

    model = nn.Sequential(MyLinear(4, 8), nn.ReLU(), MyLinear(8, 2), Scaler(2))
    print(model)
    model.eval()
    print("training flags after eval():", [m.training for m in model.modules()])
    print("buffers:", [n for n, _ in model.named_buffers()])

    # Forward hook: peek at activations without touching forward(). Debugging superpower (E14).
    h = model[0].register_forward_hook(lambda m, inp, out: print("layer0 out:", tuple(out.shape)))
    model(torch.randn(5, 4))
    h.remove()  # always remove hooks, or they fire forever


if __name__ == "__main__":
    tour()


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------
def test_mylinear_matches_nn_linear():
    torch.manual_seed(0)
    ref = nn.Linear(16, 8)
    mine = MyLinear(16, 8)
    mine.load_state_dict(ref.state_dict())  # same key names -> loads cleanly
    x = torch.randn(4, 10, 16)  # extra leading dims work too
    torch.testing.assert_close(mine(x), ref(x))


def test_same_seed_same_init_as_nn_linear():
    torch.manual_seed(42)
    ref = nn.Linear(32, 5)
    torch.manual_seed(42)
    mine = MyLinear(32, 5)
    torch.testing.assert_close(mine.weight, ref.weight)
    torch.testing.assert_close(mine.bias, ref.bias)


def test_gradients_match():
    torch.manual_seed(0)
    ref, mine = nn.Linear(6, 3), MyLinear(6, 3)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(7, 6)
    ref(x).pow(2).sum().backward()
    mine(x).pow(2).sum().backward()
    torch.testing.assert_close(mine.weight.grad, ref.weight.grad)
    torch.testing.assert_close(mine.bias.grad, ref.bias.grad)


def test_plain_tensors_and_lists_are_invisible():
    assert len(list(BrokenLinear(4, 3).parameters())) == 0
    assert len(list(MyLinear(4, 3).parameters())) == 2


def test_no_bias_and_buffers():
    lin = MyLinear(4, 3, bias=False)
    assert [n for n, _ in lin.named_parameters()] == ["weight"]
    s = Scaler(3)
    assert "running_mean" in s.state_dict() and len(list(s.parameters())) == 0
    s.eval()
    before = s.running_mean.clone()
    s(torch.randn(10, 3))
    torch.testing.assert_close(s.running_mean, before)  # eval mode: buffer frozen
