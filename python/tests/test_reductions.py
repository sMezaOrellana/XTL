import xtl
from conftest import flat, allclose


# ── sum ───────────────────────────────────────────────────────────────────────

def test_sum_axis0_shape():
    # (3, 4) -> (4,)
    assert xtl.ones([3, 4]).sum(0).shape == [4]

def test_sum_axis1_shape():
    # (3, 4) -> (3,)
    assert xtl.ones([3, 4]).sum(1).shape == [3]

def test_sum_axis0_values():
    # sum of 3 rows of ones -> 3 per column
    assert allclose(xtl.ones([3, 4]).sum(0).tolist(), [3.0] * 4)

def test_sum_axis1_values():
    # sum of 4 cols of ones -> 4 per row
    assert allclose(xtl.ones([3, 4]).sum(1).tolist(), [4.0] * 3)

def test_sum_3d_axis0_shape():
    assert xtl.ones([2, 3, 4]).sum(0).shape == [3, 4]

def test_sum_3d_axis1_shape():
    assert xtl.ones([2, 3, 4]).sum(1).shape == [2, 4]

def test_sum_3d_axis2_shape():
    assert xtl.ones([2, 3, 4]).sum(2).shape == [2, 3]

def test_sum_3d_axis2_values():
    # sum of 4 ones along last dim
    assert allclose(flat(xtl.ones([2, 3, 4]).sum(2).tolist()), [4.0] * 6)

def test_sum_1d():
    assert abs(xtl.ones([5]).sum(0).tolist() - 5.0) < 1e-5

def test_sum_after_transpose():
    a = xtl.ones([2, 3])
    a.transpose()   # shape (3, 2), non-contiguous
    c = a.sum(1)    # sum axis 1 -> shape (3,), each = 2.0
    assert c.shape == [3]
    assert allclose(c.tolist(), [2.0] * 3)
