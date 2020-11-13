import numpy as np
from lightgrad.grad import Tensor, Function

class Parameter(Tensor):
    """ Mark Tensor as a Parameter """

class Module(object):
    def __init__(self):
        self.__paramters = {}
        self.__modules = {}

    def forward(self, x):
        raise NotImplementedError()
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, val):
        if isinstance(val, Parameter):
            self.__paramters[name] = val
        if isinstance(val, Module):
            self.__modules[name] = val
        object.__setattr__(self, name, val)
    def parameters(self) -> iter:
        # yield my parameters
        yield from iter(self.__paramters.values())
        # yield parameters of submodules
        for m in self.__modules.values():
            yield from m.parameters()


""" Linear Layer """

class Linear(Module):
    def __init__(self, in_feats:int, out_feats:int, bias:bool =True):
        Module.__init__(self)
        self.w = Parameter(Tensor.xavier((in_feats, out_feats)))
        self.b = Parameter(Tensor.xavier((1, out_feats))) if bias else None
    def forward(self, x):
        return (x @ self.w + self.b) if self.b is not None else (x @ self.w)


""" Convolution """

def conv(x, w, b=None, stride:int =1):
    """ Convolution of arbitrary dimension """
    # slide window over input
    n, out_c, kernel = len(w.shape) - 1, w.shape[0], w.shape[1:]    # first dimension of kernel is in-channels
    windows = x.slide_window(kernel=kernel, strides=(stride,) * len(kernel))
    # reshape all for linear
    flat_x = windows.reshape(-1, np.prod(kernel))
    flat_w = w.transpose(*range(1, n + 1), 0).reshape(-1, out_c)    # flatten kernel and keep out-channel dimension
    flat_b = None if b is None else b.reshape(1, -1)
    # pass though linear
    linear_out = (flat_x @ flat_w + flat_b) if flat_b is not None else (flat_x @ flat_w)
    # reshape
    out_shape = windows.shape[:-2*n] + windows.shape[-2*n+1:-n] + (out_c,)
    out_permutation = tuple(range(0, len(out_shape)-n)) + (-1,) + tuple(range(len(out_shape)-n, len(out_shape)-1))
    return linear_out.reshape(*out_shape).transpose(*out_permutation)

class Conv2d(Module):
    def __init__(self, in_channels:int, out_channels:int, kernelsize:int =3, stride:int =1, pad:int =None, bias:bool =True):
        Module.__init__(self)
        self.w = Parameter(Tensor.xavier((out_channels, in_channels, kernelsize, kernelsize)))
        self.b = Parameter(Tensor.xavier((out_channels, 1, 1, 1))) if bias else None
        self.s, self.p = stride, (kernelsize // 2) if pad is None else pad
    def forward(self, x):
        return conv(x.pad(self.p) if self.p > 0 else x, self.w, self.b, stride=self.s)

