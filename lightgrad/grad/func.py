from .grads import Gradients
from typing import Tuple, Union

class __FunctionMeta(type):

    def __call__(cls, *args, **kwargs):
        # no tensors in kwargs
        assert all((not isinstance(t, Tensor) or not t.requires_grad for t in kwargs.values()))
        # create function/context instance
        f = object.__new__(cls)
        f.__init__(*args)
        # apply function
        with Gradients.no_grad():
            out_tensor = f.forward(*args, **kwargs)
            assert isinstance(out_tensor, Tensor)
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
    def parent_tensors(self) -> Tuple["Tensor"]:
        # return all parent tensors that require gradients
        return filter(lambda t: isinstance(t, Tensor) and t.requires_grad, self.__parents)

    def _set_children(self, *children):
        self.__children = children
    def _backpropagate(self, out_grad) -> Tuple["Tensor"]:
        # get all gradients and propagate backwards
        in_grads = self.backward(out_grad)
        in_grads = in_grads if isinstance(in_grads, tuple) else (in_grads,)
        assert all((isinstance(t, Tensor) for t in in_grads if t is not None)), self.__class__.__name__
        # accumulate gradients in parent tensors
        for t, g in zip(self.__parents, in_grads):
            if isinstance(t, Tensor) and t.requires_grad:
                assert g is not None
                t.add_grad(g)
        # return input gradients
        return in_grads

    def save_for_backward(ctx, *args):
        ctx.__saved_for_backward += tuple(args)
    def get_saved_tensors(self) -> Tuple["Tensor"]:
        return self.__saved_for_backward

    def forward(ctx, t:"Tensor", *args, **kwargs) -> "Tensor":
        raise NotImplementedError()
    def backward(ctx, out_grad:"Tensor") -> Union["Tensor", Tuple["Tensor"]]:
        raise RuntimeError("Cannot Backward through %s!" % ctx.__class__.__name__)

from .tensor import Tensor