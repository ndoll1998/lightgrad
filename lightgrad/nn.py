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


class Linear(Module):
    def __init__(self, in_feats:int, out_feats:int, bias:bool =True):
        Module.__init__(self)
        self.w = Parameter(Tensor.xavier((in_feats, out_feats)))
        self.b = Parameter(Tensor.xavier((1, out_feats))) if bias else None
    def forward(self, x):
        return (x @ self.w + self.b) if self.b is not None else (x @ self.w)


class Conv2d(Module):
    def __init__(self, in_channels:int, out_channels:int, kernelsize:int =3, stride:int =1, pad:int =None, bias:bool =True):
        Module.__init__(self)
        self.w = Parameter(Tensor.xavier((out_channels, in_channels, kernelsize, kernelsize)))
        self.b = Parameter(Tensor.xavier((1, out_channels, 1, 1))) if bias else None
        self.s, self.p = stride, (kernelsize // 2) if pad is None else pad
    def forward(self, x):
        y = x.__class__.conv(x.pad(self.p) if self.p > 0 else x, self.w, strides=self.s)
        y = (y + self.b) if self.b is not None else y
        return y
