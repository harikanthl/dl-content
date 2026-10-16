"""M01E01 · Tensors: what they actually are.

A tensor is five things: storage + shape + stride + dtype + device.
Storage is a flat 1-D block of bytes. Shape and stride are just a recipe
for reading that block as a grid.

We prove it by building a 2x3 tensor by hand from a flat Python list,
then checking every element against real PyTorch.
"""
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"


def contiguous_strides(shape):
    # row-major: last dim moves by 1, each earlier dim jumps over everything after it
    strides, step = [], 1
    for size in reversed(shape):
        strides.append(step)
        step *= size
    return tuple(reversed(strides))


class HandTensor:
    """Storage + shape + stride + offset. No copying, ever."""

    def __init__(self, storage, shape, stride=None, offset=0):
        self.storage = storage                          # flat python list
        self.shape = tuple(shape)
        self.stride = tuple(stride) if stride else contiguous_strides(shape)
        self.offset = offset

    def __getitem__(self, idx):
        # the one formula that IS a tensor: flat = offset + sum(i_k * stride_k)
        flat = self.offset + sum(i * s for i, s in zip(idx, self.stride))
        return self.storage[flat]

    def tolist(self):
        rows, cols = self.shape
        return [[self[(i, j)] for j in range(cols)] for i in range(rows)]


def inspect(t, name):
    print(f"{name}: shape={tuple(t.shape)} stride={t.stride()} dtype={t.dtype} "
          f"device={t.device} offset={t.storage_offset()} "
          f"storage_bytes={t.untyped_storage().nbytes()} ptr={t.data_ptr():#x}")


if __name__ == "__main__":
    flat = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    # by hand
    h = HandTensor(flat, shape=(2, 3))
    print("hand  :", h.tolist(), "stride", h.stride)

    # real pytorch
    t = torch.tensor(flat).view(2, 3)
    inspect(t, "t")
    print("storage:", t.untyped_storage().nbytes(), "bytes = 6 floats * 4 bytes")

    # same storage, different recipe: a transpose is just swapped strides
    hT = HandTensor(flat, shape=(3, 2), stride=(1, 3))
    print("handT :", hT.tolist())
    inspect(t.T, "t.T")
    print("same memory?", t.data_ptr() == t.T.data_ptr())

    # dtype changes bytes per element, device changes where the bytes live
    inspect(t.to(torch.bfloat16), "bf16")
    inspect(t.to(device), device)


# ---------- the receipt ----------
def test_contiguous_strides_match_pytorch():
    for shape in [(2, 3), (4,), (2, 3, 4), (5, 1, 7)]:
        assert contiguous_strides(shape) == torch.empty(shape).stride()


def test_hand_tensor_matches_torch():
    flat = [float(i) for i in range(1, 7)]
    t = torch.tensor(flat).view(2, 3)
    assert HandTensor(flat, (2, 3)).tolist() == t.tolist()
    # transpose: no new list, just a new stride
    assert HandTensor(flat, (3, 2), stride=(1, 3)).tolist() == t.T.tolist()
    # as_strided is PyTorch's own "give me a new recipe over the same storage"
    s = torch.as_strided(torch.tensor(flat), size=(3, 2), stride=(1, 3))
    torch.testing.assert_close(s, t.T)


def test_offset_slice_is_same_storage():
    flat = [float(i) for i in range(1, 7)]
    t = torch.tensor(flat).view(2, 3)
    row1 = t[1]                                       # a view, not a copy
    assert row1.storage_offset() == 3
    assert row1.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
    assert HandTensor(flat, (1, 3), offset=3).tolist() == [row1.tolist()]


def test_dtype_sets_bytes_per_element():
    t = torch.zeros(2, 3)
    assert t.element_size() == 4 and t.to(torch.bfloat16).element_size() == 2
    assert t.to(device).device.type == device
