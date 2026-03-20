import xtl
from conftest import flat, allclose


# ── add ───────────────────────────────────────────────────────────────────────

def test_add_same_shape():
    c = xtl.ones([2, 3]) + xtl.ones([2, 3])
    assert c.shape == [2, 3]
    assert allclose(flat(c.tolist()), [2.0] * 6)

def test_add_1d():
    c = xtl.ones([5]) + xtl.ones([5])
    assert c.shape == [5]
    assert allclose(c.tolist(), [2.0] * 5)

def test_add_broadcast_row_vector():
    c = xtl.ones([3, 4]) + xtl.ones([4])
    assert c.shape == [3, 4]
    assert allclose(flat(c.tolist()), [2.0] * 12)

def test_add_broadcast_col_vector():
    c = xtl.ones([3, 4]) + xtl.ones([3, 1])
    assert c.shape == [3, 4]
    assert allclose(flat(c.tolist()), [2.0] * 12)

def test_add_broadcast_3d_row_vector():
    c = xtl.ones([2, 3, 4]) + xtl.ones([4])
    assert c.shape == [2, 3, 4]
    assert allclose(flat(c.tolist()), [2.0] * 24)

def test_add_broadcast_3d_matrix():
    c = xtl.ones([2, 3, 4]) + xtl.ones([3, 4])
    assert c.shape == [2, 3, 4]
    assert allclose(flat(c.tolist()), [2.0] * 24)

def test_add_broadcast_leading_1():
    c = xtl.ones([2, 3, 4]) + xtl.ones([1, 3, 4])
    assert c.shape == [2, 3, 4]
    assert allclose(flat(c.tolist()), [2.0] * 24)

def test_add_commutativity():
    a, b = xtl.ones([3, 4]), xtl.ones([4])
    assert (a + b).shape == (b + a).shape
    assert allclose(flat((a + b).tolist()), flat((b + a).tolist()))

def test_add_result_ndim():
    assert (xtl.ones([2, 3, 4]) + xtl.ones([4])).ndim == 3

def test_add_rand_in_range():
    c = xtl.rand([10]) + xtl.rand([10])
    assert all(0.0 <= v <= 2.0 for v in c.tolist())


# ── sub ───────────────────────────────────────────────────────────────────────

def test_sub_same_shape():
    c = xtl.ones([2, 3]) - xtl.ones([2, 3])
    assert c.shape == [2, 3]
    assert allclose(flat(c.tolist()), [0.0] * 6)

def test_sub_broadcast():
    c = xtl.ones([3, 4]) - xtl.ones([4])
    assert c.shape == [3, 4]
    assert allclose(flat(c.tolist()), [0.0] * 12)


# ── mul ───────────────────────────────────────────────────────────────────────

def test_mul_same_shape():
    c = xtl.ones([2, 3]) * xtl.ones([2, 3])
    assert c.shape == [2, 3]
    assert allclose(flat(c.tolist()), [1.0] * 6)

def test_mul_broadcast():
    c = xtl.ones([3, 4]) * xtl.ones([4])
    assert c.shape == [3, 4]
    assert allclose(flat(c.tolist()), [1.0] * 12)


# ── div ───────────────────────────────────────────────────────────────────────

def test_div_same_shape():
    c = xtl.ones([2, 3]) / xtl.ones([2, 3])
    assert c.shape == [2, 3]
    assert allclose(flat(c.tolist()), [1.0] * 6)

def test_div_broadcast():
    c = xtl.ones([3, 4]) / xtl.ones([4])
    assert c.shape == [3, 4]
    assert allclose(flat(c.tolist()), [1.0] * 12)
