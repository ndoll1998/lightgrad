import numpy as np
from abc import ABC
from .grads import Gradients

class Tensor(ABC):

    def __init__(self, requires_grad:bool =True) -> None:
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
        raise NotImplementedError()
    def numel(self) -> int:
        return np.prod(self.shape)

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

    @classmethod
    def xavier(cls, shape, requires_grad:bool =True) -> "Tensor":
        t = cls.uniform(-1, 1, shape=shape, requires_grad=requires_grad)
        t /= np.sqrt(t.numel())
        return t

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

        node_set = {self.__ctx}
        node_list = [self.__ctx]
        # breadth-first backpropagation
        while len(node_list) > 0:
            # get current node/context
            ctx = node_list.pop(0)
            node_set.remove(ctx)
            # backpropagate and get parent contexts
            ctx._backpropagate()
            ctxs = (t.__ctx for t in ctx.parent_tensors if t.__ctx is not None)
            # add to nodes
            ctxs = set(ctx for ctx in ctxs if ctx not in node_set)
            node_list.extend(ctxs)
            node_set.update(ctxs)

    def add_grad(self, grad:"Tensor") -> None:
        # check if requires gradient
        if self.requires_grad:    
            if self.grad is None:
                self.__grad = grad.copy(requires_grad=False)
            else:
                self.__grad += grad

    def zero_grad(self, zero_graph_grads:bool =False) -> None:
        # clear my gradient
        if self.requires_grad:
            if self.grad is None:
                self.__grad = self.__class__.zeros(self.shape, requires_grad=False)
            else:
                self.__grad.data.fill(0)
        # recursivly clear gradients of all parents
        if zero_graph_grads and (self.__ctx is not None):
            for t in self.__ctx.parent_tensors:
                t.zero_grad(zero_graph_grads=True)

    @classmethod
    def register_op(cls, name:str =None, op:type =None):
        if op is not None:
            # direct use
            if not issubclass(op, Function):
                raise RuntimeError("Operators must inherit from Function! (%s)" % op.__name__)
            # not sure why this is necessary, but without dispatch wrapper
            # the op function is treatet as a static member
            dispatch = (lambda self, *args, **kwargs: op(self, *args, **kwargs))
            setattr(cls, name, dispatch)
            return op
        else:
            # use as decorator
            return lambda op: cls.register_op(name if name is not None else op.__name__, op)


# imports at bottom to avoid circular import errors
from .func import Function
from .cpu.tensor import CpuTensor
