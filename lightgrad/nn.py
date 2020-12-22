import numpy as np
from lightgrad.autograd import Tensor, AbstractTensor

class Module(object):
    def __init__(self):
        self.__paramters = {}
        self.__modules = {}

    def forward(self, x):
        raise NotImplementedError()
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, val):
        if isinstance(val, (AbstractTensor, Module)):
            self.register_param_or_module(name, val)
        object.__setattr__(self, name, val)

    def register_param_or_module(self, name, val):
        if isinstance(val, AbstractTensor):
            self.__paramters[name] = val
        elif isinstance(val, Module):
            self.__modules[name] = val
        return val
    def unregister_param_or_module(self, name):
        if name in self.__paramters:
            return self.__paramters.pop(name)
        if name in self.__modules:
            return self.__modules.pop(name)

    def parameters(self) -> iter:
        # yield my parameters
        yield from iter(self.__paramters.values())
        # yield parameters of submodules
        for m in self.__modules.values():
            yield from m.parameters()

    def named_parameters(self, prefix:str ="", separator:str =".") -> iter:
        prefix = (prefix + separator) if len(prefix) > 0 else ""
        # yield my parameters with names
        for name, p in self.__paramters.items():
            yield (prefix + name, p)
        # yield parameters with names of submodules
        for name, m in self.__modules.items():
            yield from m.named_parameters(prefix=prefix + name, separator=separator)

    def map_params(self, fn):
        # map parameters
        for key, tensor in self.__paramters.items():
            tensor = fn(tensor)
            self.__setattr__(key, tensor)
        # map sub-modules
        for m in self.__modules.values():
            m.map_params(fn)
        return self

    def load_params(self, param_dict:dict, prefix:str ="", separator:str ='.') -> None:
        param_dict = dict(param_dict)
        if len(prefix) > 0:
            prefix += separator
        # load all my parameters
        for key, p in self.__paramters.items():
            # find parameter in dict
            assert (prefix + key) in param_dict, "%s not found in param dict!" % (prefix + key)
            new_p = param_dict[prefix + key]
            # load in parameter
            if not isinstance(new_p, p.__class__):
                new_p = new_p.numpy() if isinstance(new_p, AbstractTensor) else new_p
                assert isinstance(new_p, np.ndarray), "Unexpected parameter type %s!" % new_p.__class__.__name__
                new_p = p.__class__.from_numpy(new_p)
            assert p.shape == new_p.shape, "Shapes do not align! (%s != %s)" % (p.shape, new_p.shape)
            self.__setattr__(key, new_p)
        # load sub-module parameters
        for key, m in self.__modules.items():
            m.load_params(param_dict, prefix=prefix + key, separator=separator)

class ModuleList(Module, list):
    def __init__(self, *elements):
        Module.__init__(self)
        list.__init__(self, elements)
        for i, e in enumerate(elements):
            self.register_param_or_module(str(i), e)
    def __setitem__(self, i, e):
        assert i < len(self)
        self.unregister_param_or_module(str(i))
        self.register_param_or_module(str(i), e)
        return list.__setitem__(self, i, e)
    
class Linear(Module):
    def __init__(self, in_feats:int, out_feats:int, bias:bool =True):
        Module.__init__(self)
        self.weight = Tensor.xavier((out_feats, in_feats))
        self.bias = Tensor.xavier((out_feats,)) if bias else None
    def forward(self, x):
        return (x @ self.weight.T(1, 0) + self.bias) if self.bias is not None else (x @ self.weight.T(1, 0))

class Conv2d(Module):
    def __init__(self, in_channels:int, out_channels:int, kernelsize:int =3, stride:int =1, pad:int =None, bias:bool =True):
        Module.__init__(self)
        self.w = Tensor.xavier((out_channels, in_channels, kernelsize, kernelsize))
        self.b = Tensor.xavier((1, out_channels, 1, 1)) if bias else None
        self.s, self.p = stride, (kernelsize // 2) if pad is None else pad
    def forward(self, x):
        y = (x.pad(self.p) if self.p > 0 else x).conv(self.w, strides=self.s)
        y = (y + self.b) if self.b is not None else y
        return y
