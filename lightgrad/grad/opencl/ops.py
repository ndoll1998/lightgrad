import numpy as np
import pyopencl as cl
from ..func import Function
from .tensor import OpenCLTensor
from .kernels import atom_kernel, dot_kernel

""" Helpers """

def _bi_reverse(f):
    """ reverse inputs of bi-operator """
    class F(f):
        def forward(ctx, a, b):
            return f.forward(ctx, b, a)
        def backward(ctx, out_grad):
            return reversed(f.backward(out_grad))
    return F

""" Transformations """

@OpenCLTensor.register_op()
class transpose(Function):
    def forward(ctx, a, *axes):
        assert len(axes) == len(a.shape)
        ctx.save_for_backward(axes)
        shape = tuple(a.shape[i] for i in axes)
        strides = tuple(a.strides[i] for i in axes)
        return OpenCLTensor(a.data, shape=shape, strides=strides, dtype=a.dtype)
    def backward(ctx, out_grad):
        axes, = ctx.get_saved_tensors()
        rev_axes = [None] * len(axes)
        for i, j in enumerate(axes):
            rev_axes[j] = i
        return out_grad.transpose(*rev_axes)

@OpenCLTensor.register_op()
class contiguous(Function):
    def forward(ctx, a):
        if not a.is_contiguous:
            return atom_kernel(
                a=a, out='o',
                operation_str='o = a'
            )
        return a
    def backward(ctx, out_grad):
        return out_grad

@OpenCLTensor.register_op()
class reshape(Function):
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        return OpenCLTensor(a.contiguous().data, shape=shape, dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(*shape)

""" Basic Math Operators """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        return atom_kernel(
            a=a, out="o",
            operation_str="o = -a"
        )
    def backward(ctx, out_grad):
        return -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__add__")
@OpenCLTensor.register_op("__radd__")
class add(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = a + b"
        )
    def backward(ctx, out_grad):
        return out_grad, out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__sub__")
class sub(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = a - b"
        )
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__mul__")
@OpenCLTensor.register_op("__rmul__")
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = a * b"
        )
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad * b, a * out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__div__")
class div(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = a / b"
        )
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        a_grad = atom_kernel(g=out_grad, b=b, out='o', operation_str='o = g / b')
        b_grad = atom_kernel(g=out_grad, a=a, b=b, out='o', operation_str='o = -a / pow(b, 2) * g')
        return a_grad, b_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__pow__")
class pow(Function):
    def forward(ctx, a, b):
        y = atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = pow(a, b)"
        )
        ctx.save_for_backward(a, b, y)
        return y
    def backward(ctx, out_grad):
        a, b, y = ctx.get_saved_tensors()
        a_grad = atom_kernel(g=out_grad, a=a, b=b, out='o', operation_str='o = b * pow(a, b-1) * g')
        b_grad = atom_kernel(g=out_grad, a=a, y=y, out='o', operation_str='o = g * y * log(a)')
        return a_grad, b_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__matmul__")
class dot(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return dot_kernel(a, b)    
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        a_grad = dot_kernel(out_grad, b.transpose(1, 0))
        b_grad = dot_kernel(a.transpose(1, 0), out_grad)
        return a_grad, b_grad   

# reverse operators for non-symmetrical operators
OpenCLTensor.register_op("__rsub__", _bi_reverse(sub))
OpenCLTensor.register_op("__rtruediv__", _bi_reverse(div))
OpenCLTensor.register_op("__rpow__", _bi_reverse(pow))
OpenCLTensor.register_op("__rmatmul__", _bi_reverse(dot))

""" Inplace Operators """

@OpenCLTensor.register_op('__iadd__')
class __iadd(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a += b"
        )

@OpenCLTensor.register_op('__isub__')
class __isub(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a -= b"
        )

@OpenCLTensor.register_op('__imul__')
class __imul(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a *= b"
        )
        
@OpenCLTensor.register_op('__itruediv__')
class __idiv(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a /= b"
        )

@OpenCLTensor.register_op()
class fill(Function):
    def forward(ctx, t, val):
        val = np.asarray(val, dtype=t.dtype)
        cl.enqueue_fill_buffer(t.device.queue, t.data, val, 0, t.data.size)
        return t

""" Non-Linearities """

@OpenCLTensor.register_op()
class sigmoid(Function):
    def forward(ctx, t):
        y = atom_kernel(
            t=t, out='o',
            operation_str='o = 1 / (1 + exp(-t))'
        )
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return atom_kernel(
            y=y, g=out_grad, out='o',
            operation_str='o = y * (1-y) * g'
        )

@OpenCLTensor.register_op()
class relu(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return atom_kernel(
            t=t, out='o',
            operation_str='o = (t>=0)? t : 0'
        )
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return atom_kernel(
            t=t, g=out_grad, out='o',
            operation_str='o = (t>=0)? g : 0'
        )

""" Selectors """

@OpenCLTensor.register_op("__getitem__")
class __getitem(Function):
    """ TODO: currently only supports direct item indexing (no slicing or masking). """
    def forward(ctx, a, idx):
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(a.shape)
        idx = np.asarray(idx, dtype=np.int32)
        ctx.save_for_backward(a.shape, idx)
        off = (idx * a.strides).sum() * a.dtype.itemsize
        return OpenCLTensor(a.data[off:off+a.dtype.itemsize], shape=tuple(), dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, idx = ctx.get_saved_tensors()
        grad = OpenCLTensor.zeros(shape, requires_grad=False, device=out_grad.device)
        cl.enqueue_copy(grad.device.queue, grad.data, out_grad.data, 
            byte_count=grad.dtype.itemsize,
            dest_offset=(idx * grad.strides).sum() * grad.dtype.itemsize
        )
        return grad

@OpenCLTensor.register_op("__setitem__")
class __setitem(Function):
    """ TODO: currently only supports direct item indexing (no slicing or masking). """
    def forward(ctx, a, idx, val):
        val = val if val is isinstance(val, np.ndarray) else np.asarray(val, dtype=a.dtype)
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(a.shape)
        idx = np.asarray(idx, dtype=np.int32)
        cl.enqueue_copy(a.device.queue, a.data, val, 
            device_offset=(idx * a.strides).sum() * a.dtype.itemsize
        )
        return a