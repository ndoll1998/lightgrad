import numpy as np
import pyopencl as cl
from ..func import Function
from .tensor import OpenCLTensor
from .kernels import atom_kernel, dot_kernel, reduction_kernel

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
@OpenCLTensor.register_op("T")
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
@OpenCLTensor.register_op("__truediv__")
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
class sin(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return atom_kernel(
            t=t, out='o',
            operation_str='o = sin(t)'
        )
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return atom_kernel(
            t=t, g=out_grad, out='o',
            operation_str='o = cos(t) * g'
        )

@OpenCLTensor.register_op()
class cos(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return atom_kernel(
            t=t, out='o',
            operation_str='o = cos(t)'
        )
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return atom_kernel(
            t=t, g=out_grad, out='o',
            operation_str='o = -sin(t) * g'
        )

@OpenCLTensor.register_op()
class exp(Function):
    def forward(ctx, t):
        y = atom_kernel(
            t=t, out='o',
            operation_str='o = exp(t)'
        )
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return atom_kernel(
            y=y, g=out_grad, out='o',
            operation_str='o = y * g'
        )

@OpenCLTensor.register_op()
class log(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return atom_kernel(
            t=t, out='o',
            operation_str='o = log(t)'
        )
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return atom_kernel(
            t=t, g=out_grad, out='o',
            operation_str='o = (1 / t) * g'
        )

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
class tanh(Function):
    def forward(ctx, t):
        y = atom_kernel(
            t=t, out='o',
            operation_str='o = tanh(t)'
        )
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return atom_kernel(
            y=y, g=out_grad, out='o',
            operation_str='o = (1 - y*y) * g'
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

@OpenCLTensor.register_op()
class softmax(Function):
    def forward(ctx, t, axis:int =-1):
        exps = (t - t.max(axis=axis, keepdims=True)).exp()
        return exps / exps.sum(axis=axis, keepdims=True)

""" Selectors """

def _idx_view(a, idx):
    # prepare idx tuple
    idx = (idx,) if not isinstance(idx, tuple) else idx
    idx += tuple(slice(s) for s in a.shape[len(idx):])
    idx = tuple(slice(*i.indices(s)) if isinstance(i, slice) else i for s, i in zip(a.shape, idx))
    # prepare shape and strides of sliced tensor
    shape = tuple(i.stop - i.start for i in idx if isinstance(i, slice))
    strides = tuple(st for i, st in zip(idx, a.strides) if isinstance(i, slice))
    assert len(idx) == len(a.shape) == len(a.strides)
    # get start byte position
    idx_start = np.asarray(tuple(i.start if isinstance(i, slice) else i for i in idx), dtype=np.int32)
    byte_start = (idx_start * np.asarray(a.strides, np.int32)).sum() * a.dtype.itemsize
    # create sliced tensor
    return OpenCLTensor(a.data[byte_start:], shape=shape, strides=strides, dtype=a.dtype)

@OpenCLTensor.register_op("__getitem__")
class __getitem(Function):
    """ TODO: currently only supports single index or slice (no masking) """
    def forward(ctx, a, idx):
        out = _idx_view(a, idx)
        ctx.save_for_backward(a.shape, idx)
        return out.contiguous()
    def backward(ctx, out_grad):
        # create gradient tensor
        shape, idx = ctx.get_saved_tensors()
        grad = OpenCLTensor.zeros(shape, dtype=out_grad.dtype, requires_grad=False, device=out_grad.device)
        # write output gradient to gradient according to idx
        grad[idx] = out_grad
        return grad

@OpenCLTensor.register_op("__setitem__")
class __setitem(Function):
    """ TODO: currently only supports single index or slice (no masking) """
    def forward(ctx, a, idx, val):
        atom_kernel(
            d=_idx_view(a, idx), 
            s=val, out='d',
            operation_str='d=s',
            depends_on_out=False
        )
        return a

""" Reductions """

@OpenCLTensor.register_op("sum")
class _sum(Function):
    def forward(ctx, t, axis:int =None, keepdims:bool =False):
        return reduction_kernel(t, axis=axis, keepdims=keepdims, operation_str='a + b')

@OpenCLTensor.register_op("mean")
class mean(Function):
    def forward(ctx, t, axis:int =None, keepdims:bool =False):
        # compute sum
        sum_out = t.sum(axis=axis, keepdims=keepdims)
        # divide inplace to avoid allocation of new memory
        axis = tuple(range(len(t.shape))) if axis is None else (axis,) if not isinstance(axis, tuple) else axis
        sum_out /= np.prod([t.shape[i] for i in axis])
        return sum_out
        
@OpenCLTensor.register_op()
class max(Function):
    def forward(ctx, t, axis:int =None, keepdims:bool =False):
        return reduction_kernel(t, axis=axis, keepdims=keepdims, operation_str='max(a, b)', neutral="-INFINITY")
        
@OpenCLTensor.register_op()
class min(Function):
    def forward(ctx, t, axis:int =None, keepdims:bool =False):
        return reduction_kernel(t, axis=axis, keepdims=keepdims, operation_str='min(a, b)', neutral="INFINITY")
        