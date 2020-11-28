import numpy as np
from ..tensor import Tensor

class CpuTensor(Tensor):

    def __init__(self, data:np.ndarray, dtype:type =np.float32, requires_grad:bool =True) -> None:
        # prepare data
        if isinstance(data, CpuTensor):
            data = data.data
        if isinstance(data, np.ndarray):
            if data.dtype != dtype:
                data = data.astype(dtype)
        else:
            data = np.asarray(data, dtype=dtype)
        # check data and initialize
        assert isinstance(data, np.ndarray) and (data.dtype == dtype)
        Tensor.__init__(self, data=data, requires_grad=requires_grad)

    @property
    def dtype(self):
        return self.data.dtype
    @property
    def shape(self) -> tuple:
        return self.data.shape

    def item(self):
        return self.data.item()

    @staticmethod
    def empty(shape, *args, **kwargs) -> "CpuTensor":
        return CpuTensor(np.empty(shape), *args, **kwargs)
    @staticmethod
    def zeros(shape, *args, **kwargs) -> "CpuTensor":
        return CpuTensor(np.zeros(shape), *args, **kwargs)
    @staticmethod
    def ones(shape, *args, **kwargs) -> "CpuTensor":
        return CpuTensor(np.ones(shape), *args, **kwargs)
    @staticmethod
    def uniform(low, high, shape, *args, **kwargs) -> "CpuTensor":
        return CpuTensor(np.random.uniform(low, high, size=shape), *args, **kwargs)

    def copy(self, requires_grad:bool =True) -> "CpuTensor":
        return CpuTensor(self.data.copy(), requires_grad=requires_grad)
    def numpy(self) -> np.ndarray:
        return self.data

# import operations to register them all
# import at bottom to avoid circular import errors
from . import ops