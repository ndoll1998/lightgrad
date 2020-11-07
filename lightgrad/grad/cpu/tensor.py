import numpy as np
from ..tensor import Tensor

class CpuTensor(Tensor):

    def __init__(self, data:np.ndarray, requires_grad:bool =True) -> None:
        Tensor.__init__(self, requires_grad=requires_grad)
        self.__data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> np.ndarray:
        return self.__data
    @property
    def dtype(self):
        return self.data.dtype
    @property
    def shape(self) -> tuple:
        return self.data.shape

    def item(self):
        return self.data.item()

    @staticmethod
    def zeros(shape, requires_grad:bool =True) -> "CpuTensor":
        data = np.zeros(shape).astype(np.float32)
        return CpuTensor(data, requires_grad=requires_grad)
    @staticmethod
    def ones(shape, requires_grad:bool =True) -> "CpuTensor":
        data = np.ones(shape).astype(np.float32)
        return CpuTensor(data, requires_grad=requires_grad)
    @staticmethod
    def uniform(shape, requires_grad:bool =True) -> "CpuTensor":
        data = (np.random.uniform(-1, 1, size=shape) / np.sqrt(np.prod(shape))).astype(np.float32)
        return CpuTensor(data, requires_grad=requires_grad)

    def copy(self, requires_grad:bool =True) -> "CpuTensor":
        return CpuTensor(self.data.copy(), requires_grad=requires_grad)
    def numpy(self) -> np.ndarray:
        return self.data

# import operations to register them all
# import at bottom to avoid circular import errors
from . import ops