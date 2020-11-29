import numpy as np
import pyopencl as cl
from ..tensor import Tensor

class OpenCLTensor(Tensor):
    # opencl device to use
    # this is set by the device tensor types that inherit from this type (see device.py)
    _device = None

    def __init__(self, data:cl.Buffer, shape:tuple =(-1,), dtype:type =np.float32, requires_grad:bool =True):
        # initialize tensor
        assert isinstance(data, cl.Buffer)
        Tensor.__init__(self, data=data, requires_grad=requires_grad)
        # prepare shape
        n, m = abs(np.prod(shape)), data.size // dtype.itemsize
        shape = tuple(k if k != -1 else m // n for k in shape)
        assert np.prod(shape) == m
        # save shape,dtype and device
        self.__shape = shape
        self.__dtype = dtype

    @property
    def dtype(self):
        return self.__dtype
    @property
    def shape(self) -> tuple:
        return self.__shape
    @property
    def device(self):
        return self.__class__._device

    @classmethod
    def empty(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # get device and prepare dtype
        if device is None:
            device = OpenCLDevice.default_device() if cls._device is None else cls._device
        dtype = dtype() if isinstance(dtype, type) else dtype
        # create buffer and tensor - use device tensor type
        buffer = cl.Buffer(device.ctx, cl.mem_flags.READ_WRITE, size=dtype.itemsize * np.prod(shape))
        return device.Tensor(buffer, shape=shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def from_numpy(cls, a:np.ndarray, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # get device
        if device is None:
            device = OpenCLDevice.default_device() if cls._device is None else cls._device
        # create buffer and tensor
        buffer = cl.Buffer(device.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        return device.Tensor(buffer, shape=a.shape, dtype=a.dtype, requires_grad=requires_grad)

    def numpy(self) -> np.ndarray:
        # copy buffer to numpy array
        data = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.device.queue, data, self.data)
        self.device.queue.finish()
        return data

from .device import OpenCLDevice
from . import ops