import numpy as np
import pyopencl as cl
from functools import reduce
from ..tensor import AbstractTensor

# build contiguous strides for given shape - basically just cumprod on reversed shape
_contiguous_strides = lambda shape: tuple(reduce(lambda a, x: a + [a[-1]*x] if a else [x], shape[-1:0:-1], [1,])[::-1])

class __OpenCLTensorType(AbstractTensor.__class__):
    def __call__(cls, buffer:cl.Buffer, *args, **kwargs):
        # use device tensor class from buffer context
        cls = OpenCLDevice.from_context(buffer.context).Tensor
        # create new tensor instance from arguments
        t = object.__new__(cls)
        t.__init__(buffer=buffer, *args, **kwargs)
        return t

class OpenCLTensor(AbstractTensor, metaclass=__OpenCLTensorType):
    # each opencl device has its own opencl tensor class
    # this is set in the opencldevice constructor
    __device = None

    @classmethod
    def __choose_device(cls, device):
        # define device hierarchy and choose first available device
        device_hierarchy = (device, cls.__device, OpenCLDevice.default_device())
        return next(d for d in device_hierarchy if d is not None)

    def __init__(self, buffer:cl.Buffer, shape:tuple, strides:tuple =tuple(), offset:int =0, dtype:type =np.float32, requires_grad:bool =True):
        # check buffer and initialize base tensor
        assert isinstance(buffer, (cl.Buffer, cl.tools.PooledBuffer))
        AbstractTensor.__init__(self, data=buffer, requires_grad=requires_grad)
        # save attributes
        self.__dtype = np.dtype(dtype)
        self.__shape = shape if len(shape) > 0 else (1,)
        self.__strides = strides if len(strides) > 0 else _contiguous_strides(shape)
        self.__offset = offset
        # check if shape matches buffer
        assert self.numel() <= (buffer.size // self.__dtype.itemsize) - offset, "Buffer is too small for given shape and offset! "
        assert len(self.shape) == len(self.strides), "Shapes and strides do not align! (%s <-> %s)" % (self.shape, self.strides)

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
    def offset(self) -> int:
        return self.__offset
    @property
    def device(self) -> "OpenCLDevice":
        return self.__class__.__device
        
    @classmethod
    def empty(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # get device to use
        device = cls.__choose_device(device)
        # allocate buffer and create tensor
        n = reduce(lambda x, y: x * y, shape, 1)
        buf = device.mem_pool.allocate(n * np.dtype(dtype).itemsize)
        return device.Tensor(buf, shape=shape, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        return cls.empty(shape, dtype, requires_grad, device=device).fill(0)
    @classmethod
    def ones(cls, shape:tuple, dtype:type =np.float32, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        return cls.empty(shape, dtype, requires_grad, device=device).fill(1)

    @classmethod
    def from_numpy(cls, arr:np.ndarray, requires_grad:bool =True, device:"OpenCLDevice" =None) -> "OpenCLTensor":
        # copy data from array into empty tensor
        t = cls.empty(arr.shape, dtype=arr.dtype, requires_grad=requires_grad, device=device)
        cl.enqueue_copy(t.device.queue, t.data, np.ascontiguousarray(arr)).wait()
        return t

    def numpy(self) -> np.ndarray:
        # copy data from buffer into emtpy numpy array
        arr = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.device.queue, arr, self.data, device_offset=self.offset * self.dtype.itemsize).wait()
        arr.strides = np.asarray(self.strides, dtype=np.int32) * self.dtype.itemsize
        return arr

    def copy(self, requires_grad:bool =True) -> "OpenCLTensor":
        if self.is_contiguous():
            # create empty tensor to copy data in
            o = self.device.Tensor.empty(self.shape, dtype=self.dtype, requires_grad=requires_grad)
            cl.enqueue_copy(self.device.queue, o.data, self.data, byte_count=o.numel() * self.dtype.itemsize)
            return o
        else:
            # create contiguous copy of self
            return self.contiguous(inplace=False)

    def is_contiguous(self) -> bool:
        if (len(self.strides) > 0) and (self.strides[-1] != 1):
            return False
        return not any(self.shape[i] * self.strides[i] != self.strides[i-1] for i in range(1, len(self.shape)))

    def contiguous(self, inplace=True) -> "OpenCLTensor":
        if not self.is_contiguous():
            # create contiguous copy
            cont, = atom(
                a=self, output='o',
                op='o = a'
            )
            if not inplace:
                return cont
            # apply
            self._set_data(data=cont.data)
            self.__strides = cont.strides
            self.__offset = cont.offset
        return self

from .kernels import atom
from .device import OpenCLDevice
from . import ops