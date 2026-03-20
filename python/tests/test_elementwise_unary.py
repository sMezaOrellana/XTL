import math
import xtl
from conftest import flat, allclose


# ── exp ───────────────────────────────────────────────────────────────────────

def test_exp_ones_shape():
    assert xtl.ones([2, 3]).exp().shape == [2, 3]

def test_exp_ones_values():
    # exp(1) = e
    assert allclose(flat(xtl.ones([2, 3]).exp().tolist()), [math.e] * 6)

def test_exp_zeros_values():
    # exp(0) = 1
    assert allclose(flat(xtl.zeros([4]).exp().tolist()), [1.0] * 4)


# ── sqrt ──────────────────────────────────────────────────────────────────────

def test_sqrt_shape():
    assert xtl.ones([3, 3]).sqrt().shape == [3, 3]

def test_sqrt_ones_values():
    # sqrt(1) = 1
    assert allclose(flat(xtl.ones([2, 2]).sqrt().tolist()), [1.0] * 4)


# ── neg ───────────────────────────────────────────────────────────────────────

def test_neg_shape():
    assert xtl.ones([2, 3]).neg().shape == [2, 3]

def test_neg_values():
    assert allclose(flat(xtl.ones([3]).neg().tolist()), [-1.0] * 3)


# ── log ───────────────────────────────────────────────────────────────────────

def test_log_shape():
    assert xtl.ones([2, 4]).log().shape == [2, 4]

def test_log_ones_values():
    # log(1) = 0
    assert allclose(flat(xtl.ones([5]).log().tolist()), [0.0] * 5)


# ── scalar ops ────────────────────────────────────────────────────────────────

def test_add_scalar_shape():
    assert xtl.ones([2, 3]).add_scalar(5.0).shape == [2, 3]

def test_add_scalar_values():
    assert allclose(flat(xtl.ones([4]).add_scalar(2.0).tolist()), [3.0] * 4)

def test_mul_scalar_values():
    assert allclose(flat(xtl.ones([4]).mul_scalar(3.0).tolist()), [3.0] * 4)

def test_div_scalar_values():
    assert allclose(flat(xtl.ones([4]).div_scalar(2.0).tolist()), [0.5] * 4)

def test_mul_scalar_zero():
    assert allclose(flat(xtl.ones([3]).mul_scalar(0.0).tolist()), [0.0] * 3)
