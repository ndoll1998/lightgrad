import numpy as np
from .tensor import Tensor
from .grads import Gradients

def jacobian(f, x:Tensor):
    # only allow tensor arguments for function
    assert isinstance(x, Tensor) and x.requires_grad
    # get unchanged output
    y = f(x)
    assert isinstance(y, Tensor) and y.requires_grad
    # flatten all to get jacobian matrix dimensions
    ni = x.numel()
    nj = y.numel()
    y = y.reshape(-1)
    # jacobian - use same tensor class as input
    J = x.__class__.empty((ni, nj), requires_grad=False)
    # fill jacobian
    for j in range(nj):
        # clear all gradients
        y.zero_grad(zero_graph_grads=True)
        # pick current output from tensor and backpropagate through it
        # then save gradient in jacobian matrix
        y[j].backward()
        J[:, j] = x.grad.reshape(-1)
    return J

@Gradients.no_grad()
def numerical_jacobian(f, x:Tensor, eps=1e-4):
    # only allow tensor arguments for function
    assert isinstance(x, Tensor)
    # get unchanged output
    y = f(x)
    assert isinstance(y, Tensor)
    # flatten all to get jacobian matrix dimensions
    ni = x.numel()
    nj = y.numel()
    y = y.reshape(-1)
    # jacobian - use same tensor class as input
    NJ = x.__class__.empty((ni, nj), requires_grad=False)
    # fill jacobian
    for i, idx in enumerate(np.ndindex(x.shape)):
        # build perturbation mask
        h = x.__class__.zeros(x.shape)
        h[idx] = eps
        # compute outputs
        y_add = f(x + h).reshape(-1)
        y_sub = f(x - h).reshape(-1)
        # update jacobian
        NJ[i, :] = (y_add - y_sub) / (2 * eps)
    return NJ

def gradcheck(f, x, eps=1e-4, atol=5e-4, rtol=5e-4):
    # build jacobian matrices
    J = jacobian(f, x).numpy()
    NJ = numerical_jacobian(f, x, eps).numpy()
    # compare
    return np.allclose(J, NJ, atol=atol, rtol=rtol)