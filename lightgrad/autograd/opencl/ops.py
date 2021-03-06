import numpy as np
import pyopencl as cl
from ..func import Function
from .tensor import OpenCLTensor
from . import kernels

""" Transformations """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("T")
class transpose(Function):
    def forward(ctx, a, *perm):
        assert len(a.shape) == len(perm)
        ctx.save_for_backward(perm)
        return OpenCLTensor(a.data,
            shape=tuple(a.shape[i] for i in perm),
            strides=tuple(a.strides[i] for i in perm),
            offset=a.offset,
            dtype=a.dtype
        )
    def backward(ctx, out_grad):
        perm, = ctx.get_saved_tensors()
        rev_perm = [None] * len(perm)
        for i, j in enumerate(perm):
            rev_perm[j] = i
        return out_grad.transpose(*rev_perm)

@OpenCLTensor.register_op()
class reshape(Function):
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        shape = tuple(s if s != -1 else (a.numel() // -np.prod(shape)) for s in shape)
        return OpenCLTensor(a.contiguous().data, shape=shape, dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(*shape)

""" Basic Math Operators """

@OpenCLTensor.register_op()
class neg(Function):
    def forward(ctx, a):
        return kernels.atom(
            a=a, output='o',
            op='o = -a'
        )[0]
    def backward(ctx, out_grad):
        return -out_grad

@OpenCLTensor.register_op()
class add(Function):
    def forward(ctx, a, b):
        return kernels.atom(
            a=a, b=b, output='o',
            op='o = a + b'
        )[0]
    def backward(ctx, out_grad):
        return out_grad, out_grad

@OpenCLTensor.register_op(overwrite=True)
class sub(Function):
    def forward(ctx, a, b):
        return kernels.atom(
            a=a, b=b, output='o',
            op='o = a - b'
        )[0]
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@OpenCLTensor.register_op()
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return kernels.atom(
            a=a, b=b, output='o',
            op='o = a * b'
        )[0]
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return kernels.atom(
            a=a, b=b, g=out_grad, output=('a_grad', 'b_grad'),
            op='a_grad = b * g; b_grad = a * g;'
        )

@OpenCLTensor.register_op(overwrite=True)
class div(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return kernels.atom(
            a=a, b=b, output='o',
            op="o = a / b"
        )[0]
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return kernels.atom(
            a=a, b=b, g=out_grad, output=('a_grad', 'b_grad'),
            op="a_grad = g / b; b_grad = -a / pown(b, 2) * g"
        )

@OpenCLTensor.register_op()
class pow(Function):
    def forward(ctx, a, b):
        y, = kernels.atom(
            a=a, b=b, output='o',
            op="o = pow((float)a, (float)b)"
        )
        ctx.save_for_backward(a, b, y)
        return y
    def backward(ctx, out_grad):
        a, b, y = ctx.get_saved_tensors()
        return kernels.atom(
            a=a, b=b, y=y, g=out_grad, output=('a_grad', 'b_grad'),
            op="a_grad = b * pow((float)a, (float)b-1) * g; b_grad = g * y * log(a)"
        )

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__matmul__")
class dot(Function):
    def forward(ctx, a, b):
        a_shape, b_shape = a.shape, b.shape
        out_shape = (*a_shape[:-2], a_shape[-2], b_shape[-1])
        # flatten batch dimensions
        a = a.reshape(-1, *a_shape[-2:])
        b = b.reshape(-1, *b_shape[-2:])
        ctx.save_for_backward(a, b, a_shape, b_shape)
        return kernels.dot(a, b).reshape(*out_shape)
    def backward(ctx, out_grad):
        a, b, a_shape, b_shape = ctx.get_saved_tensors()
        out_grad = out_grad.reshape(-1, *out_grad.shape[-2:])
        a_grad = kernels.dot(out_grad, b.transpose(0, 2, 1))
        b_grad = kernels.dot(a.transpose(0, 2, 1), out_grad)
        return a_grad.reshape(*a_shape), b_grad.reshape(*b_shape)

""" Inplace Operators """

@OpenCLTensor.register_op('__iadd__', overwrite=True)
class iadd(Function):
    def forward(ctx, t, other):
        return kernels.atom(
            a=t, b=other, output='a',
            op="a += b",
            additional_read=('a',)
        )[0]

@OpenCLTensor.register_op('__isub__', overwrite=True)
class isub(Function):
    def forward(ctx, t, other):
        return kernels.atom(
            a=t, b=other, output='a',
            op="a -= b",
            additional_read=('a',)
        )[0]

@OpenCLTensor.register_op('__imul__', overwrite=True)
class imul(Function):
    def forward(ctx, t, other):
        return kernels.atom(
            a=t, b=other, output='a',
            op="a *= b",
            additional_read=('a',)
        )[0]
        
@OpenCLTensor.register_op('__itruediv__', overwrite=True)
class idiv(Function):
    def forward(ctx, t, other):
        return kernels.atom(
            a=t, b=other, output='a',
            op="a /= b",
            additional_read=('a',)
        )[0]

@OpenCLTensor.register_op()
class fill(Function):
    def forward(ctx, t, val):
        val = t.dtype.type(val)
        cl.enqueue_fill_buffer(t.device.queue, t.data, val, t.offset * t.dtype.itemsize, t.numel() * t.dtype.itemsize)
        return t


""" Non-Linearities """

@OpenCLTensor.register_op()
class sin(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return kernels.atom(
            t=t, output='o',
            op='o = sin(t)'
        )[0]
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return kernels.atom(
            t=t, g=out_grad, output='o',
            op='o = cos(t) * g'
        )[0]

@OpenCLTensor.register_op()
class cos(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return kernels.atom(
            t=t, output='o',
            op='o = cos(t)'
        )[0]
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return kernels.atom(
            t=t, g=out_grad, output='o',
            op='o = -sin(t) * g'
        )[0]

@OpenCLTensor.register_op()
class exp(Function):
    def forward(ctx, t):
        y = kernels.atom(
            t=t, output='o',
            op='o = exp(t)'
        )[0]
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return kernels.atom(
            y=y, g=out_grad, output='o',
            op='o = y * g'
        )[0]

@OpenCLTensor.register_op()
class log(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return kernels.atom(
            t=t, output='o',
            op='o = log(t)'
        )[0]
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return kernels.atom(
            t=t, g=out_grad, output='o',
            op='o = (1 / t) * g'
        )[0]

@OpenCLTensor.register_op(overwrite=True)
class sigmoid(Function):
    def forward(ctx, t):
        y, = kernels.atom(
            t=t, output='o',
            op='o = 1 / (1 + exp(-t))'
        )
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return kernels.atom(
            y=y, g=out_grad, output='o',
            op='o = y * (1-y) * g'
        )[0]

@OpenCLTensor.register_op(overwrite=True)
class tanh(Function):
    def forward(ctx, t):
        y, = kernels.atom(
            t=t, output='o',
            op='o = tanh(t)'
        )
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return kernels.atom(
            y=y, g=out_grad, output='o',
            op='o = (1 - y*y) * g'
        )[0]

@OpenCLTensor.register_op()
class relu(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return kernels.atom(
            t=t, output='o',
            op='o = (t>=0)? t : 0'
        )[0]
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return kernels.atom(
            t=t, g=out_grad, output='o',
            op='o = (t>=0)? g : 0'
        )[0]

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
    # compute offset
    idx_start = np.asarray(tuple(i.start if isinstance(i, slice) else i for i in idx), dtype=np.int32)
    offset = a.offset + np.sum(idx_start * np.asarray(a.strides, np.int32))
    # create sliced tensor
    return OpenCLTensor(a.data, shape=shape, strides=strides, offset=offset, dtype=a.dtype)

@OpenCLTensor.register_op("__getitem__")
class getitem(Function):
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
class setitem(Function):
    """ TODO: currently only supports single index or slice (no masking) """
    def forward(ctx, a, idx, val):
        kernels.atom(
            d=_idx_view(a, idx), 
            s=val, output='d',
            op='d = s',
        )
        return a


""" Reductions """

def _prepare_axis(t, axis):
    axis = tuple(range(len(t.shape))) if axis is None else (axis,) if not isinstance(axis, tuple) else axis
    axis = tuple(i if i >= 0 else (len(t.shape) + i) for i in axis)
    return axis
def _squeeze(t, axis):
    assert all(t.shape[i] == 1 for i in axis)
    return t.reshape(*(s for i, s in enumerate(t.shape) if i not in axis))

@OpenCLTensor.register_op()
class sum(Function):
    def forward(ctx, x, axis:int =None, keepdims:bool =False):
        # prepare axis and apply reduction
        axis = _prepare_axis(x, axis)
        y = kernels.reduce('a + b', x, axis=axis, neutral='0')
        # save and squeeze if neccessary
        ctx.save_for_backward(x.shape, keepdims, axis)
        return y if keepdims else _squeeze(y, axis)
    def backward(ctx, out_grad):
        shape, keepdims, axis = ctx.get_saved_tensors()
        strides = out_grad.strides
        # unsqueeze
        if not keepdims:
            # build unsqueezed strides
            unsqueezed_strides, i = [], 0
            for j in range(len(shape)):
                unsqueeze_axis = (j in axis)
                unsqueezed_strides.append(0 if unsqueeze_axis else strides[i])
                i += 1 - int(unsqueeze_axis)
        else:
            # build unsqueezed strides but easier
            unsqueezed_strides = tuple(0 if i in axis else st for i, st in enumerate(strides))
        # create unsqueezed view on tensor
        return OpenCLTensor(out_grad.data, shape=shape, strides=unsqueezed_strides, dtype=out_grad.dtype)

@OpenCLTensor.register_op()
class max(Function):
    def forward(ctx, x, axis:int =None, keepdims:bool =False):
        # prepare axis and apply reduction
        axis = _prepare_axis(x, axis)
        y = kernels.reduce('max(a, b)', x, axis=axis, neutral="-INFINITY")
        # save and squeeze if neccessary
        ctx.save_for_backward(x, y)
        return y if keepdims else _squeeze(y, axis=axis)
    def backward(ctx, out_grad):
        x, y = ctx.get_saved_tensors()
        return kernels.atom(
            x=x, y=y, g=out_grad, output='o',
            op="o = (x == y)? g : 0;"
        )[0]

@OpenCLTensor.register_op()
class min(Function):
    def forward(ctx, x, axis:int =None, keepdims:bool =False):
        # prepare axis and apply reduction
        axis = _prepare_axis(x, axis)
        y = kernels.reduce('min(a, b)', x, axis=axis, neutral="INFINITY")
        # save and squeeze if neccessary
        ctx.save_for_backward(x, y)
        return y if keepdims else _squeeze(y, axis=axis)
    def backward(ctx, out_grad):
        x, y = ctx.get_saved_tensors()
        return kernels.atom(
            x=x, y=y, g=out_grad, output='o',
            op="o = (x == y)? g : 0;"
        )[0]


@OpenCLTensor.register_op()
class conv(Function):
    def forward(ctx, t, kernel, strides=1):
        # prepare strides and apply convolution
        strides = ((strides,) * (len(kernel.shape) - 2)) if isinstance(strides, int) else strides
        return kernels.conv(t, kernel, strides=strides)