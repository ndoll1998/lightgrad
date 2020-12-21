import numpy as np
from ..tensor import AbstractTensor
from ..grads import Gradients

def jacobian(f, x:AbstractTensor) -> np.ndarray:
    # only allow tensor arguments for function
    assert isinstance(x, AbstractTensor) and x.requires_grad
    # get unchanged output
    y = f(x)
    assert isinstance(y, AbstractTensor) and y.requires_grad
    # flatten all to get jacobian matrix dimensions
    ni = x.numel()
    nj = y.numel()
    y = y.reshape(-1)
    # jacobian - use same tensor class as input
    J = np.empty((ni, nj), dtype=x.dtype)
    # fill jacobian
    for j in range(nj):
        # clear all gradients
        y.zero_grad(graph_traverse=True)
        # pick current output from tensor and backpropagate through it
        # then save gradient in jacobian matrix
        y[j].backward()
        J[:, j] = x.grad.reshape(-1).numpy()
    return J

@Gradients.no_grad()
def numerical_jacobian(f, x:AbstractTensor, eps=1e-4) -> np.ndarray:
    # only allow tensor arguments for function
    assert isinstance(x, AbstractTensor)
    # get unchanged output
    y = f(x)
    assert isinstance(y, AbstractTensor)
    # flatten all to get jacobian matrix dimensions
    ni = x.numel()
    nj = y.numel()
    # jacobian - use same tensor class as input
    NJ = np.empty((ni, nj), dtype=x.dtype)
    # fill jacobian
    for i, idx in enumerate(np.ndindex(x.shape)):
        # build perturbation mask
        h = x.__class__.zeros(x.shape)
        h[idx] = eps
        # compute outputs
        y_add = f(x + h).reshape(-1)
        y_sub = f(x - h).reshape(-1)
        # update jacobian
        NJ[i, :] = (y_add - y_sub).numpy() / (2 * eps)
    return NJ

def gradcheck(f, x, eps=1e-3, atol=5e-4, rtol=5e-4):
    # build jacobian matrices
    J = jacobian(f, x)
    NJ = numerical_jacobian(f, x, eps)
    # compare
    return np.allclose(J, NJ, atol=atol, rtol=rtol)

def assert_gradcheck(f, x, eps=1e-3, atol=5e-4, rtol=5e-4):
    # build jacobian matrices
    J = jacobian(f, x)
    NJ = numerical_jacobian(f, x, eps)
    # compare
    return np.testing.assert_allclose(J, NJ, atol=atol, rtol=rtol)
