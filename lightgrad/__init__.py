from . import autograd, loss, nn, optim, data
from .autograd import Tensor, CpuTensor, Gradients, no_grad
# tensor initializer shortcuts
empty, zeros, ones = Tensor.empty, Tensor.zeros, Tensor.ones
uniform, xavier = Tensor.uniform, Tensor.xavier
from_numpy = Tensor.from_numpy