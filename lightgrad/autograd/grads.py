from functools import wraps
from collections import OrderedDict

class Gradients(object):

    __DISABLE_DEPTH = 0
    class __DisableHandler():
        __enter__ = lambda self, *args: Gradients.disable()
        __exit__ = lambda self, *args: Gradients.enable()
        def __call__(self, fn):
            @wraps(fn)
            def wrapped_fn(*args, **kwargs):
                with self: return fn(*args, **kwargs)            
            return wrapped_fn

    @staticmethod
    def disable():
        Gradients.__DISABLE_DEPTH += 1
    @staticmethod
    def enable():
        Gradients.__DISABLE_DEPTH = max(0, Gradients.__DISABLE_DEPTH - 1)
    @staticmethod
    def _is_enabled() -> bool:
        return Gradients.__DISABLE_DEPTH == 0
    @staticmethod
    def no_grad():
       return Gradients.__DisableHandler()

    @staticmethod
    def backward(ctx:"Function", grad:"AbstractTensor"):
        # nodes map contexts to their output gradient
        nodes = OrderedDict({ctx: grad})
        # breadth-first backpropagation
        while len(nodes) > 0:
            # get current node and the corresponding output gradients
            ctx, out_grad = nodes.popitem()
            # backpropagate and update parent tensor gradients
            with Gradients.no_grad():
                ctx._backpropagate(out_grad)
            # update nodes
            new_nodes = {t.ctx: t.grad for t in ctx.parent_tensors if t.requires_grad and (t.ctx is not None)}
            nodes.update(new_nodes)