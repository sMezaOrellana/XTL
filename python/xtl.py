from cffi import FFI
import os
import time as _time

ffi = FFI()

_root = os.path.join(os.path.dirname(__file__), "..")
_header_path = os.path.join(_root, "lib/tensor/tensor.h")
with open(_header_path) as f:
    _header = "\n".join(line for line in f if not line.startswith("#"))

ffi.cdef(_header)

_lib_path = os.path.join(os.path.dirname(__file__), "libxtl.so")
lib = ffi.dlopen(_lib_path)
lib.tensor_srand(int(_time.time()) & 0xFFFFFFFF)


def _make_shape(shape):
    return ffi.new("int[]", shape)


class Tensor:
    def __init__(self, ptr):
        self._ptr = ptr

    def __del__(self):
        if self._ptr is not None:
            lib.free_tensor(self._ptr)
            self._ptr = None

    @property
    def shape(self):
        return list(self._ptr.shape[0:self._ptr.ndim])

    @property
    def ndim(self):
        return self._ptr.ndim

    def __add__(self, other):
        return Tensor(lib.tensor_add(self._ptr, other._ptr))

    def __sub__(self, other):
        return Tensor(lib.tensor_sub(self._ptr, other._ptr))

    def __mul__(self, other):
        return Tensor(lib.tensor_mul(self._ptr, other._ptr))

    def __truediv__(self, other):
        return Tensor(lib.tensor_div(self._ptr, other._ptr))

    def __matmul__(self, other):
        return Tensor(lib.tensor_matmul(self._ptr, other._ptr))

    def reshape(self, shape):
        c_shape = _make_shape(shape)
        lib.tensor_reshape(self._ptr, c_shape, len(shape))
        return self

    def transpose(self):
        lib.tensor_transpose(self._ptr)
        return self

    def contiguous(self):
        return Tensor(lib.tensor_contiguous(self._ptr))

    @property
    def is_contiguous(self):
        return bool(self._ptr.contiguous)

    def sum(self, axis):
        return Tensor(lib.tensor_sum(self._ptr, axis))

    def max(self, axis):
        return Tensor(lib.tensor_max(self._ptr, axis))

    def mean(self, axis):
        return Tensor(lib.tensor_mean(self._ptr, axis))

    def var(self, axis):
        return Tensor(lib.tensor_var(self._ptr, axis))

    def softmax(self, axis):
        return Tensor(lib.tensor_softmax(self._ptr, axis))

    def layer_norm(self, axis):
        return Tensor(lib.tensor_layer_norm(self._ptr, axis))

    def gather(self, indices):
        c_indices = ffi.new("int[]", indices)
        return Tensor(lib.tensor_gather(self._ptr, c_indices, len(indices)))

    def gelu(self):
        return Tensor(lib.tensor_gelu(self._ptr))

    def gelu_apprx(self):
        return Tensor(lib.tensor_gelu_apprx(self._ptr))

    def sigmoid(self):
        return Tensor(lib.tensor_sigmoid(self._ptr))

    def tanh(self):
        return Tensor(lib.tensor_tanh(self._ptr))

    def exp(self):
        return Tensor(lib.tensor_exp(self._ptr))

    def sqrt(self):
        return Tensor(lib.tensor_sqrt(self._ptr))

    def neg(self):
        return Tensor(lib.tensor_neg(self._ptr))

    def log(self):
        return Tensor(lib.tensor_log(self._ptr))

    def add_scalar(self, s):
        return Tensor(lib.tensor_add_scalar(self._ptr, s))

    def mul_scalar(self, s):
        return Tensor(lib.tensor_mul_scalar(self._ptr, s))

    def div_scalar(self, s):
        return Tensor(lib.tensor_div_scalar(self._ptr, s))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={self.tolist()})"

    def print(self):
        import sys
        sys.stdout.flush()
        lib.print_tensor(self._ptr)

    def tolist(self):
        flat = list(self._ptr.data[0:self._ptr.size])

        if self._ptr.ndim == 0:
            return flat[0]

        def nest(data, shape):
            if len(shape) == 1:
                return data[:shape[0]]
            size = len(data) // shape[0]
            return [nest(data[i*size:(i+1)*size], shape[1:]) for i in range(shape[0])]

        return nest(flat, self.shape)


def ones(shape):
    c_shape = _make_shape(shape)
    return Tensor(lib.create_tensor_ones(c_shape, len(shape)))


def zeros(shape):
    c_shape = _make_shape(shape)
    return Tensor(lib.create_tensor_zeros(c_shape, len(shape)))


def rand(shape):
    c_shape = _make_shape(shape)
    return Tensor(lib.create_tensor_rand(c_shape, len(shape)))


def cat(tensors, axis):
    """Concatenate a list of Tensors along axis."""
    ptrs = ffi.new("Tensor*[]", [t._ptr for t in tensors])
    return Tensor(lib.tensor_cat(ptrs, len(tensors), axis))


def causal_mask(T):
    """Create a [T, T] causal attention mask."""
    return Tensor(lib.tensor_causal_mask(T))


def argmax(tensor):
    """Return the flat index of the maximum element."""
    return lib.tensor_argmax(tensor._ptr)


def sample(probs):
    """Sample an index from a 1-D probability tensor."""
    return lib.tensor_sample(probs._ptr)


def from_numpy(arr):
    """Create an XTL Tensor from a numpy array (data is copied)."""
    import numpy as np
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    shape = list(arr.shape)
    c_data = ffi.cast("float *", arr.ctypes.data)
    c_shape = _make_shape(shape)
    return Tensor(lib.tensor_from_data(c_data, c_shape, len(shape)))


def to_numpy(tensor):
    """Copy an XTL Tensor into a numpy float32 array."""
    import numpy as np
    if not tensor._ptr.contiguous:
        tensor = tensor.contiguous()
    buf = ffi.buffer(tensor._ptr.data, tensor._ptr.size * ffi.sizeof("float"))
    return np.frombuffer(buf, dtype=np.float32).copy().reshape(tensor.shape)
