import xtl
from conftest import flat, allclose


def test_2d_shape():
    assert (xtl.ones([2, 3]) @ xtl.ones([3, 4])).shape == [2, 4]

def test_2d_values():
    # ones(2,3) @ ones(3,4): each output = dot of 3 ones = 3
    assert allclose(flat((xtl.ones([2, 3]) @ xtl.ones([3, 4])).tolist()), [3.0] * 8)

def test_batched_shape():
    assert (xtl.ones([2, 2, 3]) @ xtl.ones([2, 3, 4])).shape == [2, 2, 4]

def test_batched_values():
    assert allclose(flat((xtl.ones([2, 2, 3]) @ xtl.ones([2, 3, 4])).tolist()), [3.0] * 16)

def test_broadcast_batch_shape():
    # (2,1,2,3) @ (1,3,3,4) -> (2,3,2,4)
    assert (xtl.ones([2, 1, 2, 3]) @ xtl.ones([1, 3, 3, 4])).shape == [2, 3, 2, 4]

def test_broadcast_batch_values():
    c = xtl.ones([2, 1, 2, 3]) @ xtl.ones([1, 3, 3, 4])
    assert allclose(flat(c.tolist()), [3.0] * (2 * 3 * 2 * 4))
