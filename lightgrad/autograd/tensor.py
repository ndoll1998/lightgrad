import numpy as np
from .grads import Gradients
from collections import OrderedDict

class _TensorType(type):

    def __new__(cls, name, bases, attrs):
        # create type
        T = type.__new__(cls, name, bases, attrs)
        # ignore abstract tensor type and types created during runtime
        if ('__module__' in attrs) and (attrs['__module__'] != __name__):
            # register a convert for the tensor type and register it
            backend_name = attrs['__module__'].split('.')[-2]
            Tensor.register_backend(backend_name, T)
        return T

class Tensor(metaclass=_TensorType):

    def __init__(self, data, requires_grad:bool =True) -> None:
        self.__data = data
        # gradient information
        self.__grad = None
        self.__requires_grad = requires_grad
        # expression tree context for gradient computation
        self.__ctx:Function = None

    def _set_ctx(self, ctx:"Function") -> "Tensor":
        assert isinstance(ctx, Function)
        self.__ctx = ctx
        return self
    def detach(self) -> "Tensor":
        self.__ctx = None
        return self

    @property
    def data(self):
        return self.__data
    @property
    def grad(self) -> "Tensor":
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
        return int(np.prod(self.shape))

    @staticmethod
    def empty(shape, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()
    @staticmethod
    def zeros(shape, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()
    @staticmethod
    def ones(shape, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()
    @staticmethod
    def uniform(low, high, shape, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()
    @staticmethod
    def from_numpy(a:np.ndarray, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()

    @classmethod
    def xavier(cls, shape, requires_grad:bool =True) -> "Tensor":
        t = cls.uniform(-1, 1, shape=shape, requires_grad=requires_grad)
        t /= np.sqrt(t.numel())
        return t.detach()

    def copy(self, requires_grad:bool =True) -> "Tensor":
        raise NotImplementedError()
    def numpy(self) -> np.ndarray:
        raise NotImplementedError()

    @Gradients.no_grad()
    def backward(self, allow_fill:bool =False) -> None:
        # no expression tree found
        if self.__ctx is None:
            return
        # only start backpropagation at item tensors
        if self.shape == (1,) or len(self.shape) == 0 or allow_fill:
            self.__grad = self.__class__.ones(self.shape, requires_grad=False)
        else:
            raise RuntimeError("Can only backpropagate from item tensors!")

        # nodes map contexts to their output gradient
        nodes = OrderedDict({self.__ctx: self.grad,})
        # breadth-first backpropagation
        while len(nodes) > 0:
            # get current node and the corresponding output gradients
            ctx, out_grad = nodes.popitem()
            # backpropagate and get parent contexts
            ctx._backpropagate(out_grad)
            in_tensors = tuple(t for t in ctx.parent_tensors if t.__ctx)
            # update nodes
            new_nodes = {t.__ctx: t.grad for t in ctx.parent_tensors if t.__ctx is not None}
            nodes.update(new_nodes)

    @Gradients.no_grad()
    def add_grad(self, grad:"Tensor") -> None:
        # check if requires gradient
        if self.requires_grad:
            if self.grad is None:
                self.__grad = grad.copy(requires_grad=False)
            else:
                self.__grad += grad

    def zero_grad(self, graph_traverse:bool =False) -> None:
        # clear my gradient
        if self.requires_grad:
            if self.grad is None:
                self.__grad = self.__class__.zeros(self.shape, requires_grad=False)
            else:
                self.__grad.fill(0)
        # recursivly clear gradients of all parents
        if graph_traverse and (self.__ctx is not None):
            assert self not in self.__ctx.parent_tensors
            for t in self.__ctx.parent_tensors:
                t.zero_grad(graph_traverse=True)

    @classmethod
    def register_op(cls, name:str =None, op:type =None):
        if op is not None:
            # direct use
            if not issubclass(op, Function):
                raise TypeError("Operators must inherit from Function! (%s)" % op.__name__)
            # not sure why this is necessary, but without dispatch wrapper
            # the op function is treatet as a static member
            dispatch = (lambda self, *args, **kwargs: op(self, *args, **kwargs))
            setattr(cls, name, dispatch)
            return op
        else:
            # use as decorator
            return lambda op: cls.register_op(name if name is not None else op.__name__, op)

    @staticmethod
    def register_backend(name:str, Tensor_cls:type):
        # check type
        if not issubclass(Tensor_cls, Tensor):
            raise TypeError("Backend tensors must inherit from Tensor! (%s)" % Tensor_cls.__name__)
        # create convert dispatcher
        convert = lambda t, *args, **kwargs: Tensor_cls.from_numpy(t.numpy(), *args, **kwargs)
        setattr(Tensor, name, convert)

# imports at bottom to avoid circular import errors
from .func import Function