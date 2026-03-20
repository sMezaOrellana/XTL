import math
import xtl
from conftest import flat, allclose


# ── sigmoid ───────────────────────────────────────────────────────────────────

def test_sigmoid_shape():
    assert xtl.ones([2, 3]).sigmoid().shape == [2, 3]

def test_sigmoid_at_zero():
    # sigmoid(0) = 0.5
    t = xtl.zeros([4]).sigmoid()
    assert allclose(t.tolist(), [0.5] * 4)

def test_sigmoid_output_range():
    t = xtl.rand([100]).sigmoid()
    assert all(0.0 < v < 1.0 for v in t.tolist())


# ── tanh ──────────────────────────────────────────────────────────────────────

def test_tanh_shape():
    assert xtl.ones([2, 3]).tanh().shape == [2, 3]

def test_tanh_at_zero():
    # tanh(0) = 0
    t = xtl.zeros([4]).tanh()
    assert allclose(t.tolist(), [0.0] * 4)

def test_tanh_output_range():
    t = xtl.rand([100]).tanh()
    assert all(0.0 <= v < 1.0 for v in t.tolist())


# ── gelu ──────────────────────────────────────────────────────────────────────

def test_gelu_shape():
    assert xtl.ones([2, 3]).gelu().shape == [2, 3]

def test_gelu_at_zero():
    # gelu(0) = 0
    t = xtl.zeros([4]).gelu()
    assert allclose(t.tolist(), [0.0] * 4)

def test_gelu_positive_input():
    # gelu(x) > 0 for x > 0
    t = xtl.ones([4]).gelu()
    assert all(v > 0.0 for v in t.tolist())

def test_gelu_values():
    # gelu(1) = 1 * 0.5 * (1 + erf(1/sqrt(2))) ≈ 0.8413
    t = xtl.ones([1]).gelu()
    assert allclose(t.tolist(), [0.8413], tol=1e-3)


# ── gelu_apprx ────────────────────────────────────────────────────────────────

def test_gelu_apprx_shape():
    assert xtl.ones([2, 3]).gelu_apprx().shape == [2, 3]

def test_gelu_apprx_at_zero():
    t = xtl.zeros([4]).gelu_apprx()
    assert allclose(t.tolist(), [0.0] * 4)

def test_gelu_apprx_close_to_exact():
    # approx and exact should agree within 0.001 on typical inputs
    a = xtl.ones([1])
    exact  = a.gelu().tolist()[0]
    approx = a.gelu_apprx().tolist()[0]
    assert abs(exact - approx) < 0.001
