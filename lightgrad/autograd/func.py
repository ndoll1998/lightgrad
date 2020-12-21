from .grads import Gradients
from .utils.profiler import Tracker
from typing import Tuple

class __FunctionMeta(type):
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
        with Tracker(cls.__name__), Gradients.no_grad():
            out_tensor = f.forward(*args, **kwargs)
            assert isinstance(out_tensor, AbstractTensor)
        # set context of output tensor
        if Gradients._is_enabled():
            out_tensor._set_ctx(f)
        # return
        return out_tensor

class Function(object, metaclass=__FunctionMeta):
    def __init__(self, *parents):
        self.__parents = parents
        self.__saved_for_backward = tuple()
    @property
    def parent_tensors(self) -> Tuple["AbstractTensor"]:
        # return all parent tensors that require gradients
        return filter(lambda t: isinstance(t, AbstractTensor) and t.requires_grad, self.__parents)

    def _set_children(self, *children):
        self.__children = children
    def _backpropagate(self, out_grad) -> Tuple["AbstractTensor"]:
        tensor_type = out_grad.__class__
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
                    broadcast_idx = tuple(i for i, (x, y) in enumerate(zip(t.shape, g.shape)) if x != y)
                    assert all((t.shape[i] == 1 for i in broadcast_idx)), "Cannot broadcast shapes %s and %s" % (t.shape, g.shape)
                    g = g.sum(axis=broadcast_idx, keepdims=True)
                # add gradient
                t.add_grad(g)
        # return input gradients
        return in_grads

    def save_for_backward(ctx, *args):
        ctx.__saved_for_backward += tuple(args)
    def get_saved_tensors(self) -> Tuple["AbstractTensor"]:
        return self.__saved_for_backward

    def forward(ctx, t, *args, **kwargs):
        raise NotImplementedError()
    def backward(ctx, out_grad):
        raise RuntimeError("Cannot Backward through %s!" % ctx.__class__.__name__)

from .tensor import AbstractTensor