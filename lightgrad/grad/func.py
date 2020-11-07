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
            outs = f.forward(*args, **kwargs)
            outs_tuple = outs if isinstance(outs, tuple) else (outs,)
        # set context and children
        if Gradients._is_enabled():
            outs_tuple = tuple(t._set_ctx(f) if isinstance(t, Tensor) else t for t in outs_tuple)
            f._set_children(*outs_tuple)
        # return
        return outs_tuple[0] if len(outs_tuple) == 1 else outs_tuple

class Function(object, metaclass=__FunctionMeta):

    def __init__(self, *parents):
        self.__parents = tuple(parents)
        self.__children = None
        self.__saved_for_backward = tuple()
    @property
    def parent_tensors(self) -> Tuple["Tensor"]:
        # return all parent tensors that require gradients
        return filter(lambda t: isinstance(t, Tensor) and t.requires_grad, self.__parents)

    def _set_children(self, *children):
        self.__children = tuple(children)
    def _backpropagate(self) -> Tuple["Function"]:
        assert self.__children is not None
        # get all gradients and propagate backwards
        out_grads = self.backward(*(t.grad for t in self.__children))
        out_grads = out_grads if isinstance(out_grads, tuple) else (out_grads,)
        assert all((isinstance(t, Tensor) for t in out_grads)), self.__class__.__name__
        # accumulate gradients in parent tensors
        for t, g in zip(self.__parents, out_grads):
            if isinstance(t, Tensor) and t.requires_grad:
                assert g is not None
                t.add_grad(g)

    def save_for_backward(ctx, *args):
        ctx.__saved_for_backward += tuple(args)
    def get_saved_tensors(self) -> Tuple["Tensor"]:
        return self.__saved_for_backward

    def forward(ctx, t:"Tensor", *args, **kwargs) -> Union["Tensor", Tuple["Tensor"]]:
        raise NotImplementedError()
    def backward(ctx, *out_grads:Tuple["Tensor"]) -> Union["Tensor", Tuple["Tensor"]]:
        raise RuntimeError("Cannot Backward through %s!" % ctx.__class__.__name__)

from .tensor import Tensor