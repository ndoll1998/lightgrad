import numpy as np
from .grads import Gradients
from functools import reduce

class _TensorType(type):

    def __new__(cls, name, bases, attrs):
        # create type
        T = type.__new__(cls, name, bases, attrs)
        # ignore abstract tensor type and types created during runtime
        if ('__module__' in attrs) and (attrs['__module__'] != __name__):
            # register a convert for the tensor type and register it
            backend_name = attrs['__module__'].split('.')[-2]
            AbstractTensor.register_backend(backend_name, T)
        return T

class AbstractTensor(metaclass=_TensorType):

    def __init__(self, data, requires_grad:bool =True) -> None:
        self.__data = data
        # gradient information
        self.__grad = None
        self.__requires_grad = requires_grad
        # expression tree context for gradient computation
        self.__ctx:Function = None

    def _set_ctx(self, ctx:"Function") -> "AbstractTensor":
        assert isinstance(ctx, Function) or (ctx is None)
        self.__ctx = ctx
        return self
    def _set_data(self, data) -> "AbstractTensor":
        self.__data = data
        return self

    def detach(self) -> "AbstractTensor":
        # TODO: create a copy of the tensor
        self.__ctx = None
        return self

    @property
    def ctx(self) -> "Function":
        return self.__ctx

    @property
    def data(self):
        return self.__data
    @property
    def grad(self) -> "AbstractTensor":
        return self.__grad
    @property
    def requires_grad(self) -> bool:
        return self.__requires_grad

    @property
    def dtype(self):
        raise NotImplementedError()
    @property
    def shape(self) -> tuple:
        raise NotImplementedError()

    def item(self):
        return self.numpy().item()
    def numel(self) -> int:
        return int(reduce(lambda a, b: a * b, self.shape)) if len(self.shape) > 0 else 1


    """ Initializers """

    @staticmethod
    def empty(shape, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()
    @staticmethod
    def zeros(shape, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()
    @staticmethod
    def ones(shape, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()
    @staticmethod
    def uniform(low, high, shape, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()
    @staticmethod
    def from_numpy(a:np.ndarray, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()

    @classmethod
    def xavier(cls, shape, requires_grad:bool =True) -> "AbstractTensor":
        t = cls.uniform(-1, 1, shape=shape, requires_grad=requires_grad)
        t /= np.sqrt(t.numel())
        return t.detach()

    def copy(self, requires_grad:bool =True) -> "AbstractTensor":
        raise NotImplementedError()
    def numpy(self) -> np.ndarray:
        raise NotImplementedError()


    """ Gradients """

    def backward(self, allow_fill:bool =False) -> None:
        # no expression tree found
        if self.__ctx is None:
            return
        # only start backpropagation at item tensors
        if self.shape == (1,) or len(self.shape) == 0 or allow_fill:
            self.__grad = self.__class__.ones(self.shape, requires_grad=False)
        else:
            raise RuntimeError("Can only backpropagate from item tensors!")
        # backpropagate
        Gradients.backward(self.ctx, self.grad)

    @Gradients.no_grad()
    def add_grad(self, grad:"AbstractTensor") -> None:
        # check if requires gradient
        if self.requires_grad:
            if self.grad is None:
                self.__grad = grad.copy(requires_grad=False)
            else:
                self.__grad += grad

    def zero_grad(self, traverse_graph:bool =False) -> None:
        # clear my gradient
        if self.requires_grad:
            if self.grad is None:
                self.__grad = self.__class__.zeros(self.shape, requires_grad=False)
            else:
                self.__grad.fill(0)
        # recursivly clear gradients of all parents
        if traverse_graph and (self.__ctx is not None):
            assert self not in self.__ctx.parent_tensors
            for t in self.__ctx.parent_tensors:
                t.zero_grad(traverse_graph=True)


    """ Registration of operations and backends """

    @classmethod
    def register_op(cls, name:str =None, op:type =None, overwrite:bool =False):
        if op is not None:
            # direct use
            if not issubclass(op, Function):
                raise TypeError("Operators must inherit from Function! (%s)" % op.__name__)
            # overwrite
            if not overwrite and hasattr(cls, name):
                raise RuntimeError("Function %s already registered to %s!" % (name, cls.__name__))
            # not sure why this is necessary, but without dispatch wrapper
            # the op function is treatet as a static member
            dispatch = (lambda self, *args, **kwargs: op(self, *args, **kwargs))
            setattr(cls, name, dispatch)
            return op
        else:
            # use as decorator
            return lambda op: cls.register_op(name if name is not None else op.__name__, op, overwrite=overwrite)

    @staticmethod
    def register_backend(name:str, Tensor_cls:type):
        # check type
        if not issubclass(Tensor_cls, AbstractTensor):
            raise TypeError("Backend tensors must inherit from Tensor! (%s)" % Tensor_cls.__name__)
        # create convert dispatcher
        convert = lambda t, *args, **kwargs: Tensor_cls.from_numpy(t.numpy(), *args, **kwargs)
        setattr(AbstractTensor, name, convert)

# imports at bottom to avoid circular import errors
from .func import Function
# register non-first class ops
from . import ops