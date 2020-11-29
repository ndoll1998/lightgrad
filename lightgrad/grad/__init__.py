from .grads import Gradients
from .func import Function
# import tensors
from .cpu import CpuTensor
from .opencl import OpenCLTensor
# default is cpu tensor
from .cpu.tensor import CpuTensor as Tensor