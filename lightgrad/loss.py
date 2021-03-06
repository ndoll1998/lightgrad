import numpy as np
from lightgrad.autograd import Tensor, Function

class mse(Function):
    """ Mean Squared Error """
    def forward(ctx, y, y_hat):
        err = y - y_hat
        ctx.save_for_backward(err)
        return (err ** 2).mean() / 2
    def backward(ctx, out_grad):
        err, = ctx.get_saved_tensors()
        return err * out_grad

class cross_entropy(Function):
    """ Cross Entropy Loss """
    def forward(ctx, y, y_hat, axis:int =-1):
        y = y.softmax(axis=axis)
        ctx.save_for_backward(y, y_hat, axis)
        return -y[range(y_hat.shape[0]), y_hat].log().mean()
    def backward(ctx, out_grad):
        y, y_hat, axis = ctx.get_saved_tensors()
        y[range(y_hat.shape[0]), y_hat] -= 1
        y /= y_hat.shape[0]        
        return y * out_grad
