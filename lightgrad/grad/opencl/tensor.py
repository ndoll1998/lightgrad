import numpy as np
import pyopencl as cl
from ..tensor import Tensor

def _get_strides(shape):
    strides = np.ones(len(shape), dtype=np.int32)
    for i, d in enumerate(shape[1:], 1):
        strides[:-i] *= d
    return tuple(strides)

class __OpenCLTensorType(Tensor.__class__):
    def __call__(cls, data:cl.Buffer, *args, **kwargs):
        # use device tensor class from buffer context
        cls = OpenCLDevice(data.context).Tensor
        # create new tensor instance from arguments
        t = object.__new__(cls)
        t.__init__(data=data, *args, **kwargs)
        return t

class OpenCLTensor(Tensor, metaclass=__OpenCLTensorType):
    # opencl device to use
    # this is set by the device tensor types that inherit from this type (see device.py)
    _device = None

    def __init__(self, data:cl.Buffer, shape:tuple =(-1,), strides:tuple =None, dtype:type =np.float32, requires_grad:bool =True):
        # initialize tensor
        assert isinstance(data, cl.Buffer)
        Tensor.__init__(self, data=data, requires_grad=requires_grad)
        # save data type
        self.__dtype = np.dtype(dtype)
        # prepare shape
        n, m = abs(np.prod(shape)), data.size // self.__dtype.itemsize
        shape = tuple(k if k != -1 else m // n for k in shape)
        assert np.prod(shape) == m
        # save shape and strides
        self.__shape = tuple(shape)
        self.__strides = tuple(strides) if strides is not None else _get_strides(self.__shape)
        assert len(self.__shape) == len(self.__strides), "Shape and strides don't align!"

    @property
    def dtype(self):
        return self.__dtype
    @property
    def shape(self) -> tuple:
        return self.__shape
    @property
    def strides(self) -> tuple:
        return self.__strides
    @property
    def device(self):
        return self.__class__._device

    @classmethod
    def empty(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # get device and prepare dtype
        if device is None:
            device = OpenCLDevice.default_device() if cls._device is None else cls._device
        dtype = np.dtype(dtype)
        # create buffer and tensor - use device tensor type
        buffer = cl.Buffer(device.ctx, cl.mem_flags.READ_WRITE, size=dtype.itemsize * np.prod(shape))
        return device.Tensor(buffer, shape=shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        return OpenCLTensor.empty(shape, dtype=dtype, requires_grad=requires_grad, device=device).fill(0)

    @classmethod
    def ones(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        return OpenCLTensor.empty(shape, dtype=dtype, requires_grad=requires_grad, device=device).fill(1)

    @classmethod
    def uniform(cls, low, high, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLDevice":
        a = np.random.uniform(low, high, size=shape).astype(dtype)
        return cls.from_numpy(a, requires_grad=requires_grad, device=device)

    @classmethod
    def from_numpy(cls, a:np.ndarray, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # get device
        if device is None:
            device = OpenCLDevice.default_device() if cls._device is None else cls._device
        # create buffer and tensor
        shape, dtype = a.shape, a.dtype
        strides = np.asarray(a.strides, dtype=np.uint32) // dtype.itemsize
        buffer = cl.Buffer(device.ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
        return device.Tensor(buffer, shape=shape, strides=strides, dtype=dtype, requires_grad=requires_grad)

    def numpy(self) -> np.ndarray:
        # copy buffer to numpy array
        data = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.device.queue, data, self.data)
        self.device.queue.finish()
        return data

from .device import OpenCLDevice
from . import ops