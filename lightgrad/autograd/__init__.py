from .grads import Gradients
from .func import Function
# import tensors
from .tensor import AbstractTensor
from .cpu import CpuTensor
from .opencl import OpenCLTensor, OpenCLDevice

# shortcuts
Tensor = CpuTensor      # defualt to cpu tensor
no_grad = Gradients.no_grad