import numpy as np
from ..func import Function
from .tensor import CpuTensor as Tensor
from math import ceil

""" Helpers """
_unpack = lambda t: t.data if isinstance(t, Tensor) else t

def _bi_reverse(f):
    """ reverse inputs of bi-operator """
    class F(f):
        def forward(ctx, a, b):
            return f.forward(ctx, b, a)
        def backward(ctx, out_grad):
            return reversed(f.backward(out_grad))
    return F

""" Transformations """

@Tensor.register_op()
@Tensor.register_op("T")
class transpose(Function):
    def forward(ctx, a, *axes):
        ctx.save_for_backward(axes)
        y = np.transpose(_unpack(a), axes=(axes if len(axes) > 0 else None))
        return Tensor(y)
    def backward(ctx, out_grad):
        axes, = ctx.get_saved_tensors()
        rev_axes = [None] * len(axes)
        for i, j in enumerate(axes):
            rev_axes[j] = i
        return out_grad.transpose(*rev_axes)

@Tensor.register_op()
class reshape(Function):
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        return Tensor(_unpack(a).reshape(shape))
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(*shape)

@Tensor.register_op()
class slide_window(Function):
    def forward(ctx, t, kernel, strides):
        assert len(t.shape) >= len(kernel) == len(strides) 
        t, n = _unpack(t), len(kernel)
        ctx.save_for_backward(t.shape, kernel, strides, n)
        # build shape and strides
        shape = t.shape[:-n] + tuple((d - k) // s + 1 for d, k, s in zip(t.shape[-n:], kernel, strides)) + kernel
        strides = t.strides[:-n] + tuple(ts * ws for ts, ws in zip(t.strides[-n:], strides)) + t.strides[-n:]
        # slide window
        return Tensor(np.lib.stride_tricks.as_strided(t, shape=shape, strides=strides))
    def backward(ctx, out_grad):
        # create output gradient
        in_shape, kernel, strides, n = ctx.get_saved_tensors()
        grad = Tensor.zeros(in_shape)
        # match shapes
        grad_windows = slide_window(grad, kernel, strides)
        # sum gradient of each kernel element one at a time
        # to avoid overlaps during summation
        skip_dims = tuple(slice(0, s) for s in grad_windows.shape[:-n])
        # actually add gradients in blocks of size according to strides
        # this leads to maximum non-overlapping blocks and thus minimal running time 
        # also if dimensions of shape and kernel align then there is only one block in that dimension
        strides = tuple(s if ks < d else d for s, ks, d in zip(strides, kernel, in_shape[-n:]))
        strided_shape = tuple(ceil(ks/s) for ks, s in zip(kernel, strides))
        for idx in np.ndindex(strided_shape):
            idx = skip_dims + tuple(slice(i*s, i*s+s) for i, s in zip(idx, strides))
            grad_windows[idx] += out_grad[idx]
        # return gradient tensor
        return grad

@Tensor.register_op()
class pad(Function):
    def forward(ctx, t, padding:int, dims:tuple =(-2, -1), value:float =0.0):
        ctx.save_for_backward(padding, dims)
        pad_width = np.zeros((len(t.shape), 2), dtype=np.int32)
        pad_width[dims, :] = padding
        return Tensor(np.pad(_unpack(t), pad_width=pad_width.tolist(), constant_values=value))
    def backward(ctx, out_grad):
        p, dims = ctx.get_saved_tensors()
        idx = list(slice(d) for d in out_grad.shape)
        for i in dims:
            idx[i] = slice(p, out_grad.shape[i] - p)
        return out_grad[tuple(idx)]


""" Basic Math Operators """

@Tensor.register_op()
@Tensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        return Tensor(-_unpack(a))
    def backward(ctx, out_grad):
        return -out_grad

@Tensor.register_op()
@Tensor.register_op("__add__")
@Tensor.register_op("__radd__")
class add(Function):
    def forward(ctx, a, b):
        return Tensor(_unpack(a) + _unpack(b))
    def backward(ctx, out_grad):
        return out_grad, out_grad

@Tensor.register_op()
@Tensor.register_op("__sub__")
class sub(Function):
    def forward(ctx, a, b):
        return Tensor(_unpack(a) - _unpack(b))
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@Tensor.register_op()
@Tensor.register_op("__mul__")
@Tensor.register_op("__rmul__")
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(_unpack(a) * _unpack(b))
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad * b, a * out_grad

@Tensor.register_op()
@Tensor.register_op("__truediv__")
class div(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(_unpack(a) / _unpack(b))
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad / b, -a / b**2 * out_grad

@Tensor.register_op()
@Tensor.register_op("__pow__")
class pow(Function):
    def forward(ctx, a, b):
        y = Tensor(_unpack(a) ** _unpack(b))
        ctx.save_for_backward(a, b, y)
        return y
    def backward(ctx, out_grad):
        a, b, y = ctx.get_saved_tensors()
        return b * (a ** (b - 1)) * out_grad, out_grad * y * a.log()

@Tensor.register_op()
@Tensor.register_op("__matmul__")
class dot(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(_unpack(a) @ _unpack(b))
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad @ b.T(), a.T() @ out_grad

# reverse operators for non-symmetrical operators
Tensor.register_op("__rsub__", _bi_reverse(sub))
Tensor.register_op("__rtruediv__", _bi_reverse(div))
Tensor.register_op("__rpow__", _bi_reverse(pow))
Tensor.register_op("__rmatmul__", _bi_reverse(dot))


""" Inplace Operators """

@Tensor.register_op("__iadd__")
class __iadd(Function):
    def forward(ctx, t, other):
        d = _unpack(t)
        d += _unpack(other)
        return t

@Tensor.register_op("__isub__")
class __isub(Function):
    def forward(ctx, t, other):
        d = _unpack(t)
        d -= _unpack(other)
        return t

@Tensor.register_op("__imul__")
class __imul(Function):
    def forward(ctx, t, other):
        d = _unpack(t)
        d *= _unpack(other)
        return t

@Tensor.register_op("__itruediv__")
class __itruediv(Function):
    def forward(ctx, t, other):
        d = _unpack(t)
        d /= _unpack(other)
        return t


""" Non-Linearities """

@Tensor.register_op()
class sin(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return Tensor(np.sin(_unpack(t)))
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return t.cos() * out_grad

@Tensor.register_op()
class cos(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return Tensor(np.cos(_unpack(t)))
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return -t.sin() * out_grad

@Tensor.register_op()
class exp(Function):
    def forward(ctx, t):
        y = Tensor(np.exp(_unpack(t)))
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return y * out_grad

@Tensor.register_op()
class log(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return Tensor(np.log(_unpack(t)))
    def backward(ctx, out_grad):
        x, = ctx.get_saved_tensors()
        return (1 / x) * out_grad

@Tensor.register_op()
class sigmoid(Function):
    def forward(ctx, t):
        y = Tensor(1 / (1 + np.exp(-_unpack(t))))
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return y * (1 - y) * out_grad

@Tensor.register_op()
class tanh(Function):
    def forward(ctx, t):
        y = Tensor(np.tanh(_unpack(t)))
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return (1 - y**2) * out_grad

@Tensor.register_op()
class relu(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return Tensor(np.maximum(_unpack(t), 0.0))
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return Tensor(_unpack(out_grad) * (_unpack(t) >= 0))
        # return (1 + t.exp()).log() * out_grad

@Tensor.register_op()
class softmax(Function):
    def forward(ctx, t, dim:int =-1):
        t = _unpack(t)
        exps = np.exp(t - np.max(t, axis=dim, keepdims=True))
        return Tensor(exps / np.sum(exps, axis=dim, keepdims=True))
    # TODO: backward


""" Selectors """

@Tensor.register_op("__getitem__")
class __getitem(Function):
    def forward(ctx, a, idx):
        idx = tuple(_unpack(i) for i in idx) if isinstance(idx, tuple) else _unpack(idx)
        ctx.save_for_backward(a.shape, idx)
        return Tensor(_unpack(a)[idx])
    def backward(ctx, out_grad):
        shape, idx = ctx.get_saved_tensors()
        grad = Tensor.zeros(shape, requires_grad=False)
        grad[idx] = out_grad
        return grad

@Tensor.register_op("__setitem__")
class __setitem(Function):
    def forward(ctx, a, idx, val):
        idx = tuple(_unpack(i) for i in idx) if isinstance(idx, tuple) else _unpack(idx)
        _unpack(a)[idx] = _unpack(val)
        return a


""" Reductions """

@Tensor.register_op()
class max(Function):
    def forward(ctx, x, axis:int =-1):
        x = _unpack(x)
        val = np.max(x, axis=axis)
        ctx.save_for_backward(x == val)
        return Tensor(val)
    def backward(ctx, out_grad):
        mask, = ctx.get_saved_tensors()
        return out_grad * mask

@Tensor.register_op()
class min(Function):
    def forward(ctx, x, axis:int =-1):
        x = _unpack(x)
        val = np.min(x, axis=axis)
        ctx.save_for_backward(x == val)
        return Tensor(val)
    def backward(ctx, out_grad):
        mask, = ctx.get_saved_tensors()
        return out_grad * mask

@Tensor.register_op()
class mean(Function):
    def forward(ctx, t, *args, **kwargs):
        return Tensor(_unpack(t).mean(*args, **kwargs))
    # TODO: backward

@Tensor.register_op("sum")
class _sum(Function):
    def forward(ctx, t, *args, **kwargs):
        return Tensor(_unpack(t).sum(*args, **kwargs))
    # TODO: backward


""" convolution operators """

@Tensor.register_op()
class max_pool(Function):
    def forward(ctx, t, kernelsize:tuple=(2, 2)):
        a = _unpack(t)
        n, m = len(kernelsize), len(a.shape)
        # split up pooling dimensions
        pooled_shape = sum(tuple((s//ks, ks) for s, ks in zip(a.shape[-n:], kernelsize)), tuple())
        p = a.reshape(a.shape[:-n] + pooled_shape)
        # permute dimensions to create windows
        permut_idx = tuple(range(m-n+1,m+n,2)) + tuple(range(m-n)) + tuple(range(m-n,m+n,2))
        p = p.transpose(*permut_idx)
        # flatten pooling windows
        flat_shape = (np.prod(kernelsize),) + a.shape[:-n] + tuple((s//ks for s, ks in zip(a.shape[-n:], kernelsize)))
        p = p.reshape(*flat_shape)
        # max pool and create mask for backward
        y = p.max(axis=0)
        ctx.save_for_backward(p == y, kernelsize, a.shape)
        # return tensor
        return Tensor(y)
    def backward(ctx, out_grad):
        mask, kernelsize, shape = ctx.get_saved_tensors()
        n, m = len(kernelsize), len(shape)
        # build pooling gradient
        g = mask * np.expand_dims(_unpack(out_grad), 0).repeat(np.prod(kernelsize), axis=0)
        # backward pooling windows
        g = g.reshape(*kernelsize, *g.shape[1:])
        permut_idx = tuple(range(m-n,m)) + sum(((m+i, i) for i in range(n)), tuple())
        g = g.transpose(*permut_idx).reshape(*shape)
        # return tensor
        return Tensor(g, requires_grad=False)