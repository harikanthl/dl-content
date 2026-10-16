"""M01E02 · Creating tensors and dtypes.

Creation functions, default dtypes, and the three floats every LLM uses:
float32, float16, bfloat16. Same 16 bits in fp16 and bf16, split differently:
fp16 spends them on precision, bf16 spends them on range.
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

# (exponent bits, mantissa bits)
LAYOUT = {torch.float32: (8, 23), torch.float16: (5, 10), torch.bfloat16: (8, 7)}


def float_limits(exp_bits, man_bits):
    # derive finfo from the bit layout, by hand
    bias = 2 ** (exp_bits - 1) - 1
    eps = 2.0 ** -man_bits                            # gap between 1.0 and next number
    max_ = (2 - eps) * 2.0 ** bias                    # largest exponent, all mantissa bits set
    tiny = 2.0 ** (1 - bias)                          # smallest normal number
    return eps, max_, tiny


def creation_tour():
    print(torch.zeros(2, 3).dtype, torch.ones(2, 3).dtype)          # float32
    print(torch.arange(5), torch.arange(5).dtype)                     # int64!
    print(torch.arange(0, 1, 0.25))                                   # float, end excluded
    print(torch.linspace(0, 1, 5))                                    # end included
    print(torch.randn(2, 3).dtype, torch.full((2,), 7).dtype)         # float32, int64
    print(torch.tensor([1, 2]).dtype, torch.tensor([1.0, 2]).dtype)   # int64, float32
    print((torch.tensor([1, 2]) / 2).dtype)                           # true divide -> float32
    x = torch.randn(2, 3, device=device)
    print(torch.zeros_like(x).device, torch.empty(2, 3).shape)        # *_like keeps device


def dtype_demo():
    for dt, (e, m) in LAYOUT.items():
        fi = torch.finfo(dt)
        print(f"{str(dt):<15} bits={fi.bits} exp={e} man={m} "
              f"max={fi.max:.3e} eps={fi.eps:.3e} tiny={fi.tiny:.3e}")

    # 0.1 + 0.2 bites everyone. in float32 it rounds "right" by luck. add ten 0.1s:
    print("python 0.1+0.2 == 0.3 ?", 0.1 + 0.2 == 0.3)
    s = torch.tensor(0.0)
    for _ in range(10):
        s = s + 0.1
    print(f"ten 0.1s in fp32 = {s.item():.10f}  == 1 ? {(s == 1).item()}")
    print("isclose ?", torch.isclose(s, torch.tensor(1.0)).item())   # compare floats like this

    # count to 1000 by adding 1, in bf16. it gets stuck.
    c = torch.tensor(0.0, dtype=torch.bfloat16)
    for _ in range(1000):
        c = c + 1
    print("bf16 counting to 1000 gives:", c.item())

    # overflow: fp16 dies at 65504, bf16 keeps going
    big = torch.tensor([60000.0, 70000.0])
    print("fp16:", big.half(), " bf16:", big.bfloat16())

    # precision: bf16 has 8 significant bits. 257 doesn't exist in bf16.
    print("257 in bf16:", torch.tensor(257.0).bfloat16().item())
    print("1 + 1/512 in bf16 == 1 ?", (torch.tensor(1.0).bfloat16() + 2**-9 == 1).item())

    # the classic LLM overflow: a dot product of activations
    act = torch.full((4096,), 4.0)
    print("sum of squares fp32:", (act * act).sum().item(),
          " fp16:", (act.half() * act.half()).sum().item())


if __name__ == "__main__":
    creation_tour()
    dtype_demo()


# ---------- the receipt ----------
def test_hand_derived_limits_match_finfo():
    for dt, (e, m) in LAYOUT.items():
        eps, max_, tiny = float_limits(e, m)
        fi = torch.finfo(dt)
        assert (eps, max_, tiny) == (fi.eps, fi.max, fi.tiny), dt


def test_fp16_overflows_bf16_does_not():
    x = torch.tensor(70000.0)
    assert torch.isinf(x.half())
    assert torch.isfinite(x.bfloat16())


def test_bf16_loses_precision():
    assert torch.tensor(257.0).bfloat16().item() == 256.0
    assert torch.tensor(257.0).half().item() == 257.0


def test_bf16_counter_gets_stuck_at_256():
    c = torch.tensor(0.0, dtype=torch.bfloat16)
    for _ in range(300):
        c = c + 1
    assert c.item() == 256.0          # 256 + 1 rounds back to 256: eps * 256 = 2


def test_default_dtypes():
    assert torch.arange(3).dtype == torch.int64
    assert torch.tensor([1.0]).dtype == torch.float32
    assert (torch.tensor([1, 2]) / 2).dtype == torch.float32
    assert torch.ones(2, device=device, dtype=torch.bfloat16).dtype == torch.bfloat16
