""" Non-first class operations """

from .tensor import AbstractTensor
from .func import Function, WrapperFunction
from functools import reduce

""" Basic Math Operators """

# negation operator
AbstractTensor.__neg__ = lambda t: t.neg()
# power operator
AbstractTensor.__pow__ = lambda a, b: a.pow(b)
# addition operator
AbstractTensor.__add__ = lambda a, b: a.add(b)
AbstractTensor.__iadd__ = lambda a, b: a.add(b)
AbstractTensor.__radd__ = lambda a, b: a.add(b)
# multiplication operator
AbstractTensor.__mul__ = lambda a, b: a.mul(b)
AbstractTensor.__imul__ = lambda a, b: a.mul(b)
AbstractTensor.__rmul__ = lambda a, b: a.mul(b)

@AbstractTensor.register_op()
@AbstractTensor.register_op("__sub__")
@AbstractTensor.register_op("__isub__")
@WrapperFunction.from_function
def sub(a, b):
    """ requires add and neg operator """
    return a + (-b)

@AbstractTensor.register_op()
@AbstractTensor.register_op("__truediv__")
@AbstractTensor.register_op("__itruediv__")
@WrapperFunction.from_function
def div(a, b):
    """ requires mul and pow operator """
    return a * (b ** -1)

@AbstractTensor.register_op("__rsub__")
@WrapperFunction.from_function
def rsub(b, a):
    """ requires sub operator """
    return AbstractTensor.sub(a, b)
@AbstractTensor.register_op("__rtruediv__")
@WrapperFunction.from_function
def rdiv(b, a):
    """ requires div operator """
    return AbstractTensor.div(a, b)


""" Non-Linear Activations """

@AbstractTensor.register_op()
@WrapperFunction.from_function
def sigmoid(t):
    return 1 / (1 + t.neg().exp())

@AbstractTensor.register_op()
@WrapperFunction.from_function
def tanh(t):
    return t.sigmoid() * 2 - 1

@AbstractTensor.register_op()
@WrapperFunction.from_function
def softmax(t, axis:int =-1):
    exps = (t - t.max(axis=axis, keepdims=True)).exp()
    return exps / exps.sum(axis=axis, keepdims=True)


""" Pooling """

@AbstractTensor.register_op()
class pad(Function):
    def forward(ctx, t, padding:int, dims:tuple =(-2, -1), value:float =0.0):
        n = len(dims)
        prev_pad, post_pad = padding if isinstance(padding, tuple) else (padding, padding)
        ctx.save_for_backward(prev_pad, post_pad, dims)
        # create padded tensor
        padded_shape = t.shape[:-n] + tuple(prev_pad + post_pad + s for s in t.shape[-n:])
        padded_tensor = t.__class__.empty(padded_shape, dtype=t.dtype).fill(value).detach()
        # fill values in padded tensor
        idx = tuple(slice(s) for s in t.shape[:-n]) + tuple(slice(prev_pad, prev_pad + s) for s in t.shape[-n:])
        padded_tensor[idx] = t
        return padded_tensor

    def backward(ctx, out_grad):
        prev_pad, post_pad, dims = ctx.get_saved_tensors()
        idx = list(slice(d) for d in out_grad.shape)
        for i in dims:
            idx[i] = slice(prev_pad, out_grad.shape[i] - post_pad)
        return out_grad[tuple(idx)]

@AbstractTensor.register_op()
class pool(Function):
    def forward(ctx, t, kernel:tuple =(2, 2)):
        n, m = len(kernel), len(t.shape)
        # cut input to match kernel
        in_shape = t.shape
        cut_shape = t.shape[:-n] + tuple((d//s) * s for d, s in zip(t.shape[-n:], kernel))
        cut_tensor = t[tuple(slice(d) for d in cut_shape)]
        # save for backward
        ctx.save_for_backward(kernel, cut_shape, t.shape)
        # split up pooling dimensions
        pooling_shape = sum(((s//ks, ks) for s, ks in zip(cut_tensor.shape[-n:], kernel)), tuple())
        p = cut_tensor.reshape(*cut_tensor.shape[:-n], *pooling_shape)
        # permute dimensions to create windows
        perm = tuple(range(m-n+1,m+n,2)) + tuple(range(m-n)) + tuple(range(m-n,m+n,2))
        p = p.transpose(*perm)
        # flatten pooling windows
        flat_kernel = reduce(lambda x, y: x * y, kernel)
        flat_shape = (flat_kernel,) + cut_tensor.shape[:-n] + tuple((s//ks for s, ks in zip(cut_tensor.shape[-n:], kernel)))
        return p.reshape(*flat_shape)

    def backward(ctx, out_grad):
        kernel, cut_shape, in_shape = ctx.get_saved_tensors()
        n, m = len(kernel), len(cut_shape)
        perm = tuple(range(m-n,m)) + sum(((m+i, i) for i in range(n)), tuple())
        # undo pooling windows
        cut_grad = out_grad.reshape(*kernel, *out_grad.shape[1:])
        cut_grad = cut_grad.transpose(*perm).reshape(*cut_shape)
        # pad to match input shape if neccessary
        if cut_shape != in_shape:
            grad = out_grad.__class__.zeros(in_shape)
            grad[tuple(slice(d) for d in cut_shape)] = cut_grad
            return grad
        return cut_grad

@AbstractTensor.register_op()
@WrapperFunction.from_function
def max_pool(t, kernel:tuple =(2, 2)):
    return t.pool(kernel=kernel).max(axis=0, keepdims=False)

@AbstractTensor.register_op()
@WrapperFunction.from_function
def min_pool(t, kernel:tuple =(2, 2)):
    return t.pool(kernel=kernel).min(axis=0, keepdims=False)
    
@AbstractTensor.register_op()
@WrapperFunction.from_function
def mean_pool(t, kernel:tuple =(2, 2)):
    return t.pool(kernel=kernel).mean(axis=0, keepdims=False)