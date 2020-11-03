import numpy as np
from .grads import Gradients

class Tensor(object):

    def __init__(self, data:np.ndarray, requires_grad:bool =True):
        self.__data = data.data if isinstance(data, Tensor) else data
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
    def grad(self):
        return self.__grad
    @property
    def requires_grad(self):
        return self.__requires_grad

    @property
    def dtype(self):
        return self.data.dtype
    @property
    def shape(self) -> tuple:
        return self.data.shape

    def numel(self):
        return np.prod(self.shape)
    def item(self):
        return self.data.item()

    @staticmethod
    def zeros(shape, requires_grad:bool =True):
        data = np.zeros(shape).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad)
    @staticmethod
    def ones(shape, requires_grad:bool =True):
        data = np.ones(shape).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad)
    @staticmethod
    def uniform(shape, requires_grad:bool =True):
        data = (np.random.uniform(-1, 1, size=shape) / np.sqrt(np.prod(shape))).astype(np.float32)
        return Tensor(data, requires_grad=requires_grad)

    @Gradients.no_grad()
    def backward(self, allow_fill:bool =False) -> None:
        # no expression tree found
        if self.__ctx is None:
            return
        # only start backpropagation at item tensors
        if self.shape == (1,) or len(self.shape) == 0 or allow_fill:
            self.__grad = Tensor.ones(self.shape, requires_grad=False)
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
            parent_tensors = ctx._backpropagate()
            ctxs = (t.__ctx for t in parent_tensors if t.__ctx is not None)
            # add to nodes
            ctxs = set(ctx for ctx in ctxs if ctx not in node_set)
            node_list.extend(ctxs)
            node_set.update(ctxs)

    def add_grad(self, grad:np.ndarray) -> None:
        # check if requires gradient
        if self.requires_grad:    
            if self.grad is None:
                self.__grad = Tensor(grad.data.copy(), requires_grad=False)
            else:
                self.__grad += grad

    def zero_grad(self) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.__grad = Tensor.zeros(self.shape, requires_grad=False)
            else:
                self.__grad.data.fill(0)

    @staticmethod
    def register_op(name:str =None, op:type =None):
        if op is not None:
            # direct use
            if not issubclass(op, Function):
                raise RuntimeError("Operators must inherit from Function! (%s)" % op.__name__)
            # not sure why this is necessary, but without dispatch wrapper
            # the op function is treatet as a static member
            dispatch = (lambda self, *args, **kwargs: op(self, *args, **kwargs))
            setattr(Tensor, name, dispatch)
            return op
        else:
            # use as decorator
            return lambda op: Tensor.register_op(name if name is not None else op.__name__, op)


# import operations to register them all
# import at bottom to avoid circular import errors
from .func import Function
from . import ops