import xtl
from conftest import flat, allclose


# ── reshape ───────────────────────────────────────────────────────────────────

def test_reshape_shape():
    a = xtl.ones([2, 3])
    a.reshape([6])
    assert a.shape == [6]

def test_reshape_values_preserved():
    a = xtl.ones([2, 3])
    a.reshape([3, 2])
    assert allclose(flat(a.tolist()), [1.0] * 6)

def test_reshape_to_1d():
    a = xtl.ones([2, 3, 4])
    a.reshape([24])
    assert a.shape == [24]

def test_reshape_add_dim():
    a = xtl.ones([6])
    a.reshape([2, 3])
    assert a.shape == [2, 3]

def test_reshape_contiguous_stays_true():
    a = xtl.ones([2, 3])
    a.reshape([6])
    assert a.is_contiguous


# ── transpose ─────────────────────────────────────────────────────────────────

def test_transpose_shape():
    a = xtl.ones([2, 3])
    a.transpose()
    assert a.shape == [3, 2]

def test_transpose_3d_shape():
    a = xtl.ones([2, 3, 4])
    a.transpose()
    assert a.shape == [2, 4, 3]

def test_transpose_marks_non_contiguous():
    a = xtl.ones([2, 3])
    a.transpose()
    assert not a.is_contiguous

def test_transpose_matmul():
    # ones(2,3).T @ ones(2,3) -> ones(3,2) @ ones(2,3) -> (3,3) of 2s
    a = xtl.ones([2, 3])
    b = xtl.ones([2, 3])
    a.transpose()
    c = a @ b
    assert c.shape == [3, 3]
    assert allclose(flat(c.tolist()), [2.0] * 9)


# ── contiguous ────────────────────────────────────────────────────────────────

def test_contiguous_copy_shape():
    a = xtl.ones([2, 3])
    a.transpose()
    b = a.contiguous()
    assert b.shape == [3, 2]

def test_contiguous_copy_is_contiguous():
    a = xtl.ones([2, 3])
    a.transpose()
    assert not a.is_contiguous
    b = a.contiguous()
    assert b.is_contiguous

def test_contiguous_copy_values():
    a = xtl.ones([2, 3])
    a.transpose()
    b = a.contiguous()
    assert allclose(flat(b.tolist()), [1.0] * 6)

def test_contiguous_reshape_after_contiguous():
    a = xtl.ones([2, 3])
    a.transpose()
    b = a.contiguous()
    b.reshape([6])
    assert b.shape == [6]
    assert allclose(b.tolist(), [1.0] * 6)
