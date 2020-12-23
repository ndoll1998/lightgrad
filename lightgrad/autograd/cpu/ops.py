import numpy as np
from ..func import Function
from .tensor import CpuTensor
from math import ceil

""" Helpers """

def _bi_reverse(f):
    """ reverse inputs of bi-operator """
    class F(f):
        def forward(ctx, a, b):
            return f.forward(ctx, b, a)
        def backward(ctx, out_grad):
            return reversed(f.backward(out_grad))
    return F

def _use_tensor_data(fn):
    class Fn(fn):
        def forward(ctx, *args, **kwargs):
            args = tuple(t.data if isinstance(t, CpuTensor) else t for t in args)
            kwargs = {k: t.data if isinstance(t, CpuTensor) else t for k, t in kwargs.items()}
            out_data = fn.forward(ctx, *args, **kwargs)
            return CpuTensor(data=out_data, dtype=out_data.dtype)
        def backward(ctx, out_grad):
            in_grads_data = fn.backward(ctx, out_grad.data)
            in_grads_data = in_grads_data if isinstance(in_grads_data, tuple) else (in_grads_data,)
            in_grads = tuple(CpuTensor(data=data, dtype=data.dtype) for data in in_grads_data)
            return in_grads
    Fn.__name__ = fn.__name__   # make sure to keep name as it is used for registering
    return Fn

""" Transformations """

@CpuTensor.register_op()
@CpuTensor.register_op("T")
@_use_tensor_data
class transpose(Function):
    def forward(ctx, a, *axes):
        ctx.save_for_backward(axes)
        return np.transpose(a, axes=(axes if len(axes) > 0 else None))
    def backward(ctx, out_grad):
        axes, = ctx.get_saved_tensors()
        rev_axes = [None] * len(axes)
        for i, j in enumerate(axes):
            rev_axes[j] = i
        return out_grad.transpose(*rev_axes)

@CpuTensor.register_op()
@_use_tensor_data
class reshape(Function):
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        return a.reshape(shape)
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(shape)


""" Basic Math Operators """

@CpuTensor.register_op()
@CpuTensor.register_op("__neg__")
@_use_tensor_data
class neg(Function):
    def forward(ctx, a):
        return -a
    def backward(ctx, out_grad):
        return -out_grad

@CpuTensor.register_op()
@CpuTensor.register_op("__add__")
@CpuTensor.register_op("__radd__")
@_use_tensor_data
class add(Function):
    def forward(ctx, a, b):
        return a + b
    def backward(ctx, out_grad):
        return out_grad, out_grad

@CpuTensor.register_op()
@CpuTensor.register_op("__sub__")
@_use_tensor_data
class sub(Function):
    def forward(ctx, a, b):
        return a - b
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@CpuTensor.register_op()
@CpuTensor.register_op("__mul__")
@CpuTensor.register_op("__rmul__")
@_use_tensor_data
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a * b
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad * b, a * out_grad

@CpuTensor.register_op()
@CpuTensor.register_op("__truediv__")
@_use_tensor_data
class div(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a / b
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad / b, -a / b**2 * out_grad

@CpuTensor.register_op()
@CpuTensor.register_op("__pow__")
@_use_tensor_data
class pow(Function):
    def forward(ctx, a, b):
        y = a ** b
        ctx.save_for_backward(a, b, y)
        return y
    def backward(ctx, out_grad):
        a, b, y = ctx.get_saved_tensors()
        return b * (a ** (b - 1)) * out_grad, out_grad * y * np.log(a)

@CpuTensor.register_op()
@CpuTensor.register_op("__matmul__")
@_use_tensor_data
class dot(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a @ b
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad @ b.T, a.T @ out_grad

# reverse operators for non-symmetrical operators
CpuTensor.register_op("__rsub__", _bi_reverse(sub))
CpuTensor.register_op("__rtruediv__", _bi_reverse(div))
CpuTensor.register_op("__rpow__", _bi_reverse(pow))
CpuTensor.register_op("__rmatmul__", _bi_reverse(dot))


""" Inplace Operators """

@CpuTensor.register_op("__iadd__")
@_use_tensor_data
class __iadd(Function):
    def forward(ctx, t, other):
        t += other
        return t

@CpuTensor.register_op("__isub__")
@_use_tensor_data
class __isub(Function):
    def forward(ctx, t, other):
        t -= other
        return t

@CpuTensor.register_op("__imul__")
@_use_tensor_data
class __imul(Function):
    def forward(ctx, t, other):
        t *= other
        return t

@CpuTensor.register_op("__itruediv__")
@_use_tensor_data
class __itruediv(Function):
    def forward(ctx, t, other):
        t /= other
        return t

@CpuTensor.register_op()
@_use_tensor_data
class fill(Function):
    def forward(ctx, t, val):
        t.fill(val)
        return t


""" Non-Linearities """

@CpuTensor.register_op()
@_use_tensor_data
class sin(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return np.sin(t)
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return np.cos(t) * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class cos(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return np.cos(t)
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return -np.sin(t) * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class exp(Function):
    def forward(ctx, t):
        y = np.exp(t)
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return y * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class log(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return np.log(t)
    def backward(ctx, out_grad):
        x, = ctx.get_saved_tensors()
        return (1 / x) * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class sigmoid(Function):
    def forward(ctx, t):
        y = 1 / (1 + np.exp(-t))
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return y * (1 - y) * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class tanh(Function):
    def forward(ctx, t):
        y = np.tanh(t)
        ctx.save_for_backward(y)
        return y
    def backward(ctx, out_grad):
        y, = ctx.get_saved_tensors()
        return (1 - y**2) * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class relu(Function):
    def forward(ctx, t):
        ctx.save_for_backward(t)
        return np.maximum(t, 0.0)
    def backward(ctx, out_grad):
        t, = ctx.get_saved_tensors()
        return out_grad * (t >= 0)
        # return (1 + t.exp()).log() * out_grad

@CpuTensor.register_op()
@_use_tensor_data
class softmax(Function):
    def forward(ctx, t, axis:int =-1):
        exps = np.exp(t - np.max(t, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)
    # TODO: backward


""" Selectors """

@CpuTensor.register_op("__getitem__")
@_use_tensor_data
class __getitem(Function):
    def forward(ctx, a, idx):
        if isinstance(idx, tuple):
            idx = tuple(t.data if isinstance(t, CpuTensor) else t for t in idx)
        ctx.save_for_backward(a.shape, idx)
        return a[idx]
    def backward(ctx, out_grad):
        shape, idx = ctx.get_saved_tensors()
        grad = CpuTensor.zeros(shape, requires_grad=False)
        grad[idx] = out_grad
        return grad

@CpuTensor.register_op("__setitem__")
@_use_tensor_data
class __setitem(Function):
    def forward(ctx, a, idx, val):
        if isinstance(idx, tuple):
            idx = tuple(t.data if isinstance(t, CpuTensor) else t for t in idx)
        a[idx] = val
        return a


""" Reductions """

@CpuTensor.register_op()
@_use_tensor_data
class max(Function):
    def forward(ctx, x, *args, **kwargs):
        val = np.max(x, *args, **kwargs)
        ctx.save_for_backward(x == val)
        return val
    def backward(ctx, out_grad):
        mask, = ctx.get_saved_tensors()
        return out_grad * mask

@CpuTensor.register_op()
@_use_tensor_data
class min(Function):
    def forward(ctx, x, *args, **kwargs):
        val = np.min(x, *args, **kwargs)
        ctx.save_for_backward(x == val)
        return val
    def backward(ctx, out_grad):
        mask, = ctx.get_saved_tensors()
        return out_grad * mask

@CpuTensor.register_op()
@_use_tensor_data
class mean(Function):
    def forward(ctx, t, *args, **kwargs):
        return t.mean(*args, **kwargs)
    # TODO: backward

@CpuTensor.register_op("sum")
@_use_tensor_data
class _sum(Function):
    def forward(ctx, t, *args, **kwargs):
        return t.sum(*args, **kwargs)
    # TODO: backward


""" convolution operators """

@CpuTensor.register_op()
@_use_tensor_data
class conv(Function):
    @staticmethod
    def __stride(t, kernel_shape, strides):
        n = len(kernel_shape)
        shape = t.shape[:-n] + tuple((d - k) // s + 1 for d, k, s in zip(t.shape[-n:], kernel_shape, strides)) + kernel_shape
        strides = t.strides[:-n] + tuple(ts * ws for ts, ws in zip(t.strides[-n:], strides)) + t.strides[-n:]
        return np.lib.stride_tricks.as_strided(t, shape=shape, strides=strides)

    def forward(ctx, t, kernel, strides):
        # preparation
        n, m = len(kernel.shape) - 1, len(t.shape)
        strides = ((strides,) * n) if isinstance(strides, int) else strides
        assert m >= n == len(strides)
        # build shape and strides
        x = conv.__stride(t, kernel.shape[1:], strides)
        # reduce to matrix multiplication
        flat_x = x.reshape(-1, np.prod(kernel.shape[1:]))
        flat_w = kernel.reshape(kernel.shape[0], -1)
        y = flat_x @ flat_w.T
        # save for backward
        ctx.save_for_backward(flat_x, flat_w, t.shape, kernel.shape, strides)
        # reverse flatten for output
        y = y.reshape(*x.shape[:-n], -1)
        y = y.swapaxes(-n-1, -1).squeeze(-1)
        return y

    def backward(ctx, out_grad):
        # preparation
        flat_x, flat_w, in_shape, w_shape, strides = ctx.get_saved_tensors()

        # flatten output gradient
        n = len(w_shape) - 1
        flat_out_grad = np.moveaxis(out_grad, -n, -1).reshape(-1, w_shape[0])
        # dot product backward
        flat_x_grad = flat_out_grad @ flat_w
        flat_w_grad = flat_out_grad.T @ flat_x
        # reshape kernel gradient
        w_grad = flat_w_grad.reshape(w_shape)

        # build input gradient
        x_grad = np.zeros(in_shape)
        # create windows of x-grad
        x_grad_windows = conv.__stride(x_grad, w_shape[1:], strides)
        out_x_grad_windows = flat_x_grad.reshape(x_grad_windows.shape)
        # sum gradient of each kernel element one at a time
        # to avoid overlaps during summation
        skip_dims = tuple(slice(0, s) for s in x_grad_windows.shape[:-n])
        # actually add gradients in blocks of size according to strides
        # this leads to maximum non-overlapping blocks and thus minimal running time 
        # also if dimensions of shape and kernel align then there is only one block in that dimension
        strides = tuple(s if ks < d else d for s, ks, d in zip(strides, w_shape[1:], in_shape[-n:]))
        strided_shape = tuple(ceil(ks/s) for ks, s in zip(w_shape[1:], strides))
        for idx in np.ndindex(strided_shape):
            idx = skip_dims + tuple(slice(i*s, i*s+s) for i, s in zip(idx, strides))
            x_grad_windows[idx] += out_x_grad_windows[idx]
        # return
        return x_grad, w_grad

@CpuTensor.register_op()
@_use_tensor_data
class max_pool(Function):
    def forward(ctx, a, kernelsize:tuple=(2, 2)):
        n, m = len(kernelsize), len(a.shape)
        # cut input to match stride
        in_shape = a.shape
        cut_shape = a.shape[:-n] + tuple((d//s) * s for d, s in zip(a.shape[-n:], kernelsize))
        a = a[tuple(slice(d) for d in cut_shape)]
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
        ctx.save_for_backward(p == y, kernelsize, in_shape, cut_shape)
        # return tensor
        return y
    def backward(ctx, out_grad):
        mask, kernelsize, in_shape, cut_shape = ctx.get_saved_tensors()
        n, m = len(kernelsize), len(cut_shape)
        # build pooling gradient
        g = mask * np.expand_dims(out_grad, 0).repeat(np.prod(kernelsize), axis=0)
        # backward pooling windows
        g = g.reshape(*kernelsize, *g.shape[1:])
        permut_idx = tuple(range(m-n,m)) + sum(((m+i, i) for i in range(n)), tuple())
        g = g.transpose(*permut_idx).reshape(*cut_shape)
        # pad to match input shape
        return np.pad(g, [(0, s-c) for s, c in zip(in_shape, cut_shape)])

@CpuTensor.register_op()
@_use_tensor_data
class pad(Function):
    def forward(ctx, t, padding:int, dims:tuple =(-2, -1), value:float =0.0):
        ctx.save_for_backward(padding, dims)
        pad_width = np.zeros((len(t.shape), 2), dtype=np.int32)
        pad_width[dims, :] = padding
        return np.pad(t, pad_width=pad_width.tolist(), constant_values=value)
    def backward(ctx, out_grad):
        p, dims = ctx.get_saved_tensors()
        idx = list(slice(d) for d in out_grad.shape)
        for i in dims:
            idx[i] = slice(p, out_grad.shape[i] - p)
        return out_grad[tuple(idx)]