from .grads import Gradients
from .utils.profiler import Tracker
from typing import Tuple

class __FunctionType(type):
    @staticmethod
    @Gradients.no_grad()
    def apply(f, *args, **kwargs):
        return f.forward(*args, **kwargs)
        
    def __call__(cls, *args, **kwargs):
        # no tensors in kwargs
        assert all((not isinstance(t, AbstractTensor) or not t.requires_grad for t in kwargs.values()))
        # create function/context instance
        f = object.__new__(cls)
        f.__init__(*args)
        # check tensors
        tensors = tuple(t for t in list(args) + list(kwargs.values()) if isinstance(t, AbstractTensor))
        tensor_type = tensors[0].__class__
        assert all((isinstance(t, tensor_type) for t in tensors[1:])), "All Tensors must be of the same type! %s" % str(tuple(t.__class__.__name__ for t in tensors))
        # apply function
        with Tracker(cls.__name__):
            out_tensor = cls.__class__.apply(f, *args, **kwargs)
            assert isinstance(out_tensor, AbstractTensor)
        # set context of output tensor
        if Gradients._is_enabled():
            out_tensor._set_ctx(f)
        # return
        return out_tensor

class Function(object, metaclass=__FunctionType):
    def __init__(self, *parents):
        self.__parents = parents
        self.__saved_for_backward = tuple()
    @property
    def parent_tensors(self) -> Tuple["AbstractTensor"]:
        # return all parent tensors that require gradients
        return filter(lambda t: isinstance(t, AbstractTensor) and t.requires_grad, self.__parents)

    def _backpropagate(self, out_grad):
        # propagate backwards
        with Tracker(self.__class__.__name__, backward=True):
            in_grads = self.backward(out_grad)
            in_grads = in_grads if isinstance(in_grads, tuple) else (in_grads,)
            # accumulate gradients in parent tensors
            for t, g in zip(self.__parents, in_grads):
                if isinstance(t, AbstractTensor) and t.requires_grad:
                    assert g is not None
                    # reverse broadcast gradient shape
                    if g.shape != t.shape:
                        assert len(g.shape) >= len(t.shape), "Cannot unbroadcast shapes %s and %s" % (t.shape, g.shape)
                        d = len(g.shape) - len(t.shape)
                        broadcast_idx = tuple(range(d))
                        broadcast_idx += tuple(d + i for i, (x, y) in enumerate(zip(t.shape, g.shape[d:])) if x != y)
                        g = g.sum(axis=broadcast_idx, keepdims=True)
                        g = g.reshape(*g.shape[d:])
                    assert g.shape == t.shape
                    # add gradient
                    t.add_grad(g)

    def save_for_backward(ctx, *args):
        ctx.__saved_for_backward += tuple(args)
    def get_saved_tensors(self) -> Tuple["AbstractTensor"]:
        return self.__saved_for_backward

    def forward(ctx, t, *args, **kwargs):
        raise NotImplementedError()
    def backward(ctx, out_grad):
        raise RuntimeError("Cannot Backward through %s!" % ctx.__class__.__name__)

class __WrapperFunctionType(__FunctionType):
    @staticmethod
    def apply(f, *args, **kwargs):
        # apply with enabled gradients to track internal expression tree
        out_tensor = f.forward(*args, **kwargs)
        # save internal expression tree
        # the context of the output tensor will be overridden with the function f
        f._set_internal_ctx(out_tensor.ctx)
        # return output
        return out_tensor

class WrapperFunction(Function, metaclass=__WrapperFunctionType):
    """ Function that computes gradients implicitly by backpropagating through the expression tree """
    def __init__(self, *parents):
        Function.__init__(self, *parents)
        # internal expression tree
        self.__internal_ctx = None
    def _set_internal_ctx(self, ctx:Function):
        self.__internal_ctx = ctx
    def _backpropagate(self, out_grad) -> Tuple["AbstractTensor"]:
        with Tracker(self.__class__.__name__, backward=True):
            # mark parent tensors as borders of propagation
            parent_context = [(p, p.ctx) for p in self.parent_tensors]
            [p._set_ctx(None) for p, _ in parent_context]
            # backpropagate
            Gradients.backward(self.__internal_ctx, out_grad)
            # unmark parent tensors
            [p._set_ctx(ctx) for p, ctx in parent_context]
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def from_function(fn):
        # decorator to create wrapper function class from python function
        dispatch = lambda ctx, *args, **kwargs: fn(*args, **kwargs)
        return type(fn.__name__, (WrapperFunction,), {'forward': dispatch})

from .tensor import AbstractTensor