"""M01E03 · Indexing, views and strides.

Slicing returns a VIEW: same storage, new shape/stride/offset.
Transpose only swaps strides. .view() needs strides it can reuse; .reshape()
copies when it has to. Fancy indexing always copies.
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"


def same_storage(a, b):
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def show(t, name):
    print(f"{name:<14} shape={str(tuple(t.shape)):<9} stride={str(t.stride()):<8} "
          f"offset={t.storage_offset()} contiguous={t.is_contiguous()}")


# from-scratch versions using only as_strided: (size, stride, offset) over the SAME storage
def my_transpose(t):
    return t.as_strided(size=t.shape[::-1], stride=t.stride()[::-1], storage_offset=t.storage_offset())


def my_column(t, j):
    # column j of a 2-D contiguous tensor: walk down rows, start j elements in
    return t.as_strided(size=(t.shape[0],), stride=(t.stride(0),), storage_offset=t.storage_offset() + j)


def my_every_other_row(t):
    return t.as_strided(size=((t.shape[0] + 1) // 2, t.shape[1]),
                        stride=(2 * t.stride(0), t.stride(1)), storage_offset=t.storage_offset())


if __name__ == "__main__":
    t = torch.arange(12.0).view(3, 4)
    show(t, "t")
    show(t[1], "t[1]")              # offset 4
    show(t[:, 1], "t[:, 1]")        # stride 4, offset 1
    show(t[::2], "t[::2]")          # stride (8, 1)
    show(t.T, "t.T")                # stride (1, 4), not contiguous

    # views share memory: writing through one changes the other
    col = t[:, 1]
    col[0] = 100.0
    print("t[0, 1] after writing col:", t[0, 1].item())

    # the failure case
    try:
        t.T.view(12)
    except RuntimeError as e:
        print("view fails:", str(e).split(".")[0])
    r = t.T.reshape(12)             # works, but silently copies
    print("reshape shares storage?", same_storage(r, t))
    print("contiguous() on contiguous is free?", t.contiguous() is t)
    print("fancy index t[[0, 2]] shares storage?", same_storage(t[[0, 2]], t))
    print("mask t[t > 5] shares storage?", same_storage(t[t > 5], t))


# ---------- the receipt ----------
def test_my_views_match_pytorch_and_share_storage():
    t = torch.arange(12.0, device=device).view(3, 4)
    for mine, ref in [(my_transpose(t), t.T), (my_column(t, 1), t[:, 1]),
                      (my_every_other_row(t), t[::2])]:
        torch.testing.assert_close(mine, ref)
        assert mine.stride() == ref.stride()
        assert same_storage(mine, t) and same_storage(ref, t)


def test_view_vs_reshape_on_transpose():
    t = torch.arange(12.0).view(3, 4)
    try:
        t.T.view(12)
        raise AssertionError("view should have failed")
    except RuntimeError:
        pass
    r = t.T.reshape(12)
    assert not same_storage(r, t)                  # reshape had to copy
    assert same_storage(t.reshape(4, 3), t)        # but on contiguous input it's a free view
    torch.testing.assert_close(r, t.T.contiguous().view(12))


def test_what_copies():
    t = torch.arange(12.0).view(3, 4)
    assert t.contiguous() is t                     # no copy when already contiguous
    assert not same_storage(t.T.contiguous(), t)   # copy when not
    assert not same_storage(t[[0, 2]], t)          # advanced indexing copies
    assert not same_storage(t[t > 5], t)           # boolean mask copies
