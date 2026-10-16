# EP010 · M01E08 · nn.Module and nn.Parameter

**Module:** M01 PyTorch from Zero · **Hardware:** Mac (MPS) · **Target length:** 10–15 min
**Code:** `code/M01-pytorch/e08_nn_module_and_nn_parameter.py` · **Explainer:** `explainers/scenes/m01e08_nn_module_and_nn_parameter.py`

---

## 🎬 Hook (first 15 seconds, face cam)
> "nn.Linear is about fifteen lines of Python. I'm going to write it from scratch, load
> PyTorch's weights into mine, and get the exact same numbers. Then you'll never be scared
> of a Module again."

## ❓ The question (why this matters, 45 s)
In E07 we trained linear regression with raw tensors. We kept a list of `w` and `b` by hand,
passed them to the update by hand, and zeroed them by hand. That works for two tensors.
GPT-2 small has 148 tensors. Llama 3 8B has 291. Nobody tracks those by hand.

`nn.Module` is the bookkeeping. Every model in this series, from the bigram model to the
full Llama block, is a tree of Modules. `model.parameters()` is how the optimizer finds the
weights. `model.state_dict()` is what a checkpoint is. `model.eval()` is how dropout knows
to switch off. If you understand what Module does on assignment, you understand all three.

## 🧸 Intuition (whiteboard / explainer clip, 2 min)
> **ELI5: a Module is a backpack with a label maker.**
> When you put something in the backpack (`self.weight = ...`), the backpack checks the
> label. If it says "Parameter", it writes it on the inventory list. If it says "Module",
> it adds that whole smaller backpack to the list. Anything else, a plain tensor, a Python
> list, it lets you carry, but it doesn't write it down.
>
> The optimizer only reads the inventory list. So if it isn't on the list, it never learns.

The one picture: a tree. `Sequential` at the root, `MyLinear` / `ReLU` / `MyLinear` as
children, `weight` and `bias` as leaves. `parameters()` walks the leaves.
`state_dict()` walks the leaves plus buffers and gives each a dotted name: `0.weight`,
`2.bias`, `3.running_mean`.

**How it actually works (one sentence for the nerds):** `nn.Module.__setattr__` is
overridden. It checks `isinstance(value, Parameter)` and `isinstance(value, Module)` and
files the value into `self._parameters` or `self._modules`. That's the magic. There is no
other magic.

## 🪤 The trap
```python
self.weight = torch.randn(out, in, requires_grad=True)   # looks fine. is not.
self.layers = [nn.Linear(d, d) for _ in range(4)]        # also not.
```
Both run. Both produce outputs. The loss even computes. But `model.parameters()` is empty
(or missing the list), so the optimizer has nothing to update, `state_dict()` doesn't save
it, and `.to("mps")` doesn't move it. You get a model that "trains" with a flat loss, or a
device-mismatch error three files away.

The fix: `nn.Parameter(...)` for tensors, `nn.ModuleList([...])` for lists of layers,
`nn.ModuleDict` for dicts. The code has a `BrokenLinear` that does both wrong; print its
`parameters()` on camera. It prints `[]`.

## 📐 The math (derive on screen, 2–3 min)
1. Linear layer: `y = x Wᵀ + b`
   - `x`: `(B, in)`, `W`: `(out, in)`, `b`: `(out,)`, `y`: `(B, out)`.
   - Why is W stored `(out, in)` and not `(in, out)`? Each row is one output neuron's
     weights. It's the convention from `torch.nn.functional.linear`, and we copy it so
     `load_state_dict` works between the two classes. Say this, people always ask.
2. It works for any leading dims: `(B, T, in) @ (in, out) -> (B, T, out)`. This is
   exactly the shape every transformer projection uses later.
3. Init: `kaiming_uniform_(W, a=√5)` turns out to be `U(-1/√in, 1/√in)`.
   Derivation: gain = `√(2/(1+a²)) = √(2/6) = 1/√3`, bound = `gain·√(3/in) = 1/√in`.
   The bias uses the same bound. The point of `1/√in`: each output is a sum of `in`
   terms, so its variance grows with `in`; scaling by `1/√in` keeps it O(1).
4. Param count: `in·out + out`. `MyLinear(4, 3)` has 15. `nn.Linear(784, 512)` has 401,920.

## 💻 Code it (from scratch, 4–5 min)
```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()                    # sets up _parameters/_modules dicts. forget it -> AttributeError
        self.weight = nn.Parameter(torch.empty(out_features, in_features))  # registered on assignment
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):               # copy nn.Linear's init exactly
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.shape[1])
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                     # (..., in) -> (..., out)
        y = x @ self.weight.T
        if self.bias is not None:
            y = y + self.bias                 # (out,) broadcasts over leading dims
        return y
```
Then the tour (in `tour()`):
- `named_parameters()` → `[('weight', (3, 4)), ('bias', (3,))]`
- `state_dict().keys()` → `['weight', 'bias']`, same keys as `nn.Linear`. That's why the receipt works.
- `BrokenLinear(4, 3).named_parameters()` → `[]`. Pause on this.
- `register_buffer("running_mean", ...)`: saved and moved, never trained. BatchNorm's
  running stats and, later, RoPE's cos/sin cache and the causal mask live here.
- `model.eval()` flips `.training` to False on every submodule, recursively.
- `register_forward_hook`: peek at a layer's output without editing `forward`. We use this
  for real in E14. Remove it when done.

Call `model(x)`, never `model.forward(x)`. `__call__` is what runs the hooks.

## ✅ The receipt (1 min)
```python
ref = nn.Linear(16, 8)
mine = MyLinear(16, 8)
mine.load_state_dict(ref.state_dict())          # PyTorch's weights into my class
torch.testing.assert_close(mine(x), ref(x))     # x is (4, 10, 16): leading dims too
```
Plus three more in the file: same seed gives the identical init as `nn.Linear` (we made
the same RNG calls in the same order), gradients match after `.backward()`, and the broken
class has zero parameters. `pytest -q` → **5 passed**.

## 📊 Measure (1 min)
Measured on M4 Mac mini, CPU, `(4096, 1024) -> 1024`, mean of 50 forwards:
`nn.Linear` 5.1 ms, `MyLinear` 4.8 ms. Same speed within noise. `nn.Linear` calls a fused
`addmm`; we do matmul then add. For layers this size, the matmul is the whole cost.
The lesson: the built-in isn't faster because of secret sauce. It's the same math.

## ⚠️ Gotchas
- **Plain tensor attribute instead of `nn.Parameter`.** Silent. `parameters()` is empty,
  loss stays flat. Check: `sum(p.numel() for p in model.parameters())` against your
  hand count. Do this for every new model in the series.
- **Python list of layers.** Same bug, different costume. Use `nn.ModuleList`. A classic
  symptom: `RuntimeError: Tensor for argument weight is on cpu but expected on mps`
  after `model.to("mps")`, because the listed layers never moved (measured, torch 2.14).
- **Forgot `super().__init__()`.** Error is loud and confusing:
  `AttributeError: cannot assign module before Module.__init__() call`.
- **Weight shape `(in, out)`.** Your own layer works, but `load_state_dict` from
  `nn.Linear` fails with `size mismatch for weight: copying a param with shape
  torch.Size([8, 16]) ... current model is torch.Size([16, 8])`.
- **Calling `.forward()` directly.** Skips hooks. Works until it doesn't.
- **Buffer vs attribute.** A mask stored as `self.mask = torch.tril(...)` won't follow
  `.to(device)`. `register_buffer` it. Pass `persistent=False` if you don't want it saved.

## 🎤 Interview lines
- *"A Module is a container that auto-registers Parameters and child Modules on
  attribute assignment, via `__setattr__`. Everything else, `parameters()`, `state_dict()`,
  `.to()`, `train()/eval()`, is just a recursive walk over that registry."*
- *"Parameters are trained, buffers are saved and moved but not trained. BatchNorm
  running stats are buffers. So is a causal mask."*
- *"nn.Linear stores weight as (out, in), and default init is Kaiming-uniform with a=√5,
  which is U(±1/√fan_in). The 1/√fan_in keeps the output variance from growing with width."*
- If pushed on why `model(x)` not `model.forward(x)`: *"`__call__` runs pre- and
  post-forward hooks around `forward`. Calling forward directly bypasses them."*

## 🔗 Next
Now that the optimizer can find the weights, we need something to optimize and something
to do the optimizing. Next, M01E09: we write cross-entropy and SGD-with-momentum by hand and
match `torch.optim.SGD` bit for bit.

## 📹 Metadata
- **YouTube title:** `nn.Linear from scratch (and the bug that makes your model not learn) | PyTorch from Zero #8`
- **Thumbnail (3–5 words):** `IT'S NOT A PARAMETER`
- **Short (which beat, 45–60 s):** The trap. `BrokenLinear` prints `parameters() = []`, loss flat, fix with one word.
- **X post:** "Your model trains, loss computes, nothing learns. 9 times out of 10: you stored a weight as a plain tensor, not nn.Parameter. Module only registers what it recognizes. Wrote nn.Linear from scratch to show it, matches PyTorch exactly."
- **Sources:** PyTorch documentation, `torch.nn.Module` and `torch.nn.Linear` source (https://docs.pytorch.org/docs/stable/index.html)
