import pyopencl as cl
from typing import Union
from .tensor import OpenCLTensor

_DEVICE_MAP = {
    cl.device_type.ACCELERATOR: 'ACCELERATOR', 
    cl.device_type.CUSTOM:      'CUSTOM', 
    cl.device_type.DEFAULT:     'DEFAULT', 
    cl.device_type.ALL:         'ALL', 
    cl.device_type.CPU:         'CPU', 
    cl.device_type.GPU:         'GPU'
}

class OpenCLDevice(object):
    # default device
    __default_device = None
    # all populated contexts and queues
    __contexts = {}
    __queues = {}     # queues are unique for each context
    # all populated device tensor types
    __tensor_types = {}

    def __init__(self, context_or_deviceId:Union[int, cl.Context], device_type:int =cl.device_type.ALL):
        # create or find context
        if isinstance(context_or_deviceId, int):
            # get device
            device_list = sum((p.get_devices(device_type) for p in cl.get_platforms()), [])
            device = device_list[context_or_deviceId]
            # find or create context
            if device in OpenCLDevice.__contexts:
                ctx = OpenCLDevice.__contexts[device]
                queue = OpenCLDevice.__queues[device]
                tensor_type = OpenCLDevice.__tensor_types[device]
            else:
                # not yet populated context
                ctx = cl.Context([device])
                queue = cl.CommandQueue(ctx)
                # create device tensor type
                name = "%s(%s:%i)" % (OpenCLTensor.__name__, _DEVICE_MAP[device.type], len(OpenCLDevice.__contexts))
                tensor_type = type(name, (OpenCLTensor,), {'_device': self})
                # store
                OpenCLDevice.__contexts[device] = ctx
                OpenCLDevice.__queues[device] = cl.CommandQueue(ctx)
                OpenCLDevice.__tensor_types[device] = tensor_type
        else:
            # We assume that the provided context was populated
            # and thus also queue and tensor-type
            ctx = context_or_deviceId
            queue = OpenCLDevice.__queues[device]
            tensor_type = OpenCLDevice.__tensor_types[device]
        # save context and queue
        self.__tensor_type = tensor_type
        self.__ctx = ctx
        self.__queue = queue

    @property
    def ctx(self) -> cl.Context:
        return self.__ctx
    @property
    def queue(self) -> cl.CommandQueue:
        return self.__queue
    @property
    def Tensor(self) -> type:
        return self.__tensor_type

    @staticmethod
    def any(device_type:int =cl.device_type.ALL) -> "OpenCLDevice":
        return OpenCLDevice(0, device_type=cl.device_type.ALL)

    @staticmethod
    def default_device() -> "OpenCLDevice":
        if OpenCLDevice.__default_device is None:
            device = OpenCLDevice.any()
            OpenCLDevice.set_default_device(device)
        return OpenCLDevice.__default_device
    @staticmethod
    def set_default_device(device:"OpenCLDevice"):
        OpenCLDevice.__default_device = device