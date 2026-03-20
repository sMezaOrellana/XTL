import xtl
from conftest import flat, allclose


def test_ones_shape():
    assert xtl.ones([2, 3]).shape == [2, 3]

def test_ones_values():
    assert allclose(flat(xtl.ones([2, 3]).tolist()), [1.0] * 6)

def test_zeros_shape():
    assert xtl.zeros([2, 3]).shape == [2, 3]

def test_zeros_values():
    assert allclose(flat(xtl.zeros([2, 3]).tolist()), [0.0] * 6)

def test_rand_shape():
    assert xtl.rand([3, 4]).shape == [3, 4]

def test_rand_values_in_range():
    assert all(0.0 <= v <= 1.0 for v in xtl.rand([100]).tolist())

def test_ndim():
    assert xtl.ones([2, 3, 4]).ndim == 3
