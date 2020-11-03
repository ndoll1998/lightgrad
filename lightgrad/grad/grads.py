from functools import wraps

class Gradients(object):

    __DISABLE_DEPTH = 0
    class __DisableHandler():
        def __enter__(self, *args):
            Gradients.disable()
        def __exit__(self, *args):
            Gradients.enable()
        def __call__(self, fn):
            @wraps(fn)
            def wrapped_fn(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)            
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
    def no_grad(fn=None):
        return Gradients.__DisableHandler()

