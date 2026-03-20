import math
import xtl
from conftest import flat, allclose


# ── max ───────────────────────────────────────────────────────────────────────

def test_max_axis0_shape():
    assert xtl.ones([3, 4]).max(0).shape == [4]

def test_max_axis1_shape():
    assert xtl.ones([3, 4]).max(1).shape == [3]

def test_max_axis0_values():
    # all ones, max = 1
    assert allclose(xtl.ones([3, 4]).max(0).tolist(), [1.0] * 4)

def test_max_3d_axis1_shape():
    assert xtl.ones([2, 3, 4]).max(1).shape == [2, 4]


# ── mean ──────────────────────────────────────────────────────────────────────

def test_mean_axis0_shape():
    assert xtl.ones([3, 4]).mean(0).shape == [4]

def test_mean_axis1_shape():
    assert xtl.ones([3, 4]).mean(1).shape == [3]

def test_mean_axis0_values():
    assert allclose(xtl.ones([3, 4]).mean(0).tolist(), [1.0] * 4)

def test_mean_axis1_values():
    assert allclose(xtl.ones([3, 4]).mean(1).tolist(), [1.0] * 3)

def test_mean_1d():
    assert abs(xtl.ones([6]).mean(0).tolist() - 1.0) < 1e-5


# ── var ───────────────────────────────────────────────────────────────────────

def test_var_constant_tensor():
    # var of all-ones = 0
    assert allclose(xtl.ones([3, 4]).var(1).tolist(), [0.0] * 3)

def test_var_axis0_shape():
    assert xtl.ones([3, 4]).var(0).shape == [4]

def test_var_axis1_shape():
    assert xtl.ones([3, 4]).var(1).shape == [3]


# ── softmax ───────────────────────────────────────────────────────────────────

def test_softmax_shape():
    assert xtl.ones([2, 4]).softmax(1).shape == [2, 4]

def test_softmax_sums_to_one():
    out = xtl.ones([3, 5]).softmax(1)
    rows = out.tolist()
    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-5

def test_softmax_uniform_input():
    # uniform input -> uniform output = 1/N
    N = 4
    out = xtl.ones([2, N]).softmax(1)
    rows = out.tolist()
    for row in rows:
        assert allclose(row, [1.0 / N] * N)

def test_softmax_axis0_shape():
    assert xtl.ones([4, 3]).softmax(0).shape == [4, 3]

def test_softmax_axis0_sums_to_one():
    out = xtl.ones([4, 3]).softmax(0)
    cols = [[out.tolist()[r][c] for r in range(4)] for c in range(3)]
    for col in cols:
        assert abs(sum(col) - 1.0) < 1e-5


# ── layer_norm ────────────────────────────────────────────────────────────────

def test_layer_norm_shape():
    assert xtl.ones([2, 4]).layer_norm(1).shape == [2, 4]

def test_layer_norm_constant_input_values():
    # constant input -> all zeros after centering (std=0+eps -> near-zero output)
    out = xtl.ones([2, 4]).layer_norm(1)
    vals = flat(out.tolist())
    assert allclose(vals, [0.0] * 8, tol=1e-2)

def test_layer_norm_mean_near_zero():
    # after layer_norm the mean along axis should be ~0
    import xtl
    a = xtl.ones([3, 8])
    # use rand to get varied values — but we have no easy way to construct
    # from Python, so just verify shape and that constant input -> ~0
    out = a.layer_norm(1)
    rows = out.tolist()
    for row in rows:
        assert abs(sum(row) / len(row)) < 1e-3


# ── gather ────────────────────────────────────────────────────────────────────

def test_gather_shape():
    a = xtl.ones([5, 3])
    out = a.gather([0, 2, 4])
    assert out.shape == [3, 3]

def test_gather_values():
    a = xtl.ones([5, 3])
    out = a.gather([1, 3])
    assert allclose(flat(out.tolist()), [1.0] * 6)

def test_gather_single_row():
    a = xtl.ones([4, 2])
    out = a.gather([2])
    assert out.shape == [1, 2]
    assert allclose(out.tolist()[0], [1.0, 1.0])

def test_gather_repeated_index():
    a = xtl.ones([3, 4])
    out = a.gather([0, 0, 0])
    assert out.shape == [3, 4]
    assert allclose(flat(out.tolist()), [1.0] * 12)
