import pyopencl as cl

_DEVICE_MAP = {
    cl.device_type.ACCELERATOR: 'ACCELERATOR', 
    cl.device_type.CUSTOM:      'CUSTOM', 
    cl.device_type.DEFAULT:     'DEFAULT', 
    cl.device_type.ALL:         'ALL', 
    cl.device_type.CPU:         'CPU', 
    cl.device_type.GPU:         'GPU'
}

class OpenCLDevicePool(object):

    def __init__(self, device_cls):
        # gather all devices
        try:
            # raises error if no opencl platform was found
            cl_plts = cl.get_platforms()
            cl_devices = sum([p.get_devices(cl.device_type.ALL) for p in cl_plts], [])
        except:
            print("No OpenCL Platforms available!")
        # all populated devices
        self.__cl_devices = cl_devices
        self.__devices = {cl_device: None for cl_device in cl_devices}
        self.__device_cls = device_cls

    def get_cl_device_id(self, cl_device:cl.Device, local=False):
        device_type = cl_device.get_info(cl.device_info.TYPE) if local else cl.device_type.ALL
        cl_devices = self.get_all_cl_device(device_type)
        return cl_devices.index(cl_device)
    def get_all_cl_device(self, device_type:cl.device_type =cl.device_type.ALL) -> list:
        return tuple(p for p in self.__cl_devices
            if device_type in (p.get_info(cl.device_info.TYPE), cl.device_type.ALL))
    def get_cl_device(self, device_type:cl.device_type, device_id:int):
        # get cl device
        cl_devices = self.get_all_cl_device(device_type=device_type)
        return cl_devices[device_id] if device_id < len(cl_devices) else None
    def get_device_from_cl_device(self, cl_device:cl.Device):
        # find device
        device = self.__devices[cl_device]
        # create new device if not populated yet
        if device is None:
            # prevent recursive call of constructor
            device = object.__new__(self.__device_cls)
            device.__init__(cl_device)
            self.__devices[cl_device] = device
        return device

class __OpenCLDeviceType(type):

    def __new__(cls, name, bases, attrs):
        T = type.__new__(cls, name, bases, attrs)
        # initialize device pool
        T.device_pool = OpenCLDevicePool(T)
        return T

    def __call__(cls, device_type:cl.device_type =cl.device_type.ALL, device_id:int =0):
        # get cl-device
        cl_device = cls.device_pool.get_cl_device(device_type, device_id)
        # handle unvailable devices
        if cl_device is None:
            raise RuntimeError("OpenCLDevice not found!")
        # find device in pool
        return cls.device_pool.get_device_from_cl_device(cl_device)

class OpenCLDevice(metaclass=__OpenCLDeviceType):
    # default devices and device pool
    __default_device = None
    device_pool = None

    def __init__(self, cl_device:cl.Device):
        self.__cl_device = cl_device
        self.__desc = "%s:%i" % (
            _DEVICE_MAP[cl_device.get_info(cl.device_info.TYPE)], 
            OpenCLDevice.device_pool.get_cl_device_id(cl_device, local=True)
        )
        # create context and queue
        self.__context = cl.Context(devices=[cl_device])
        self.__queue = cl.CommandQueue(self.__context, device=cl_device)
        # create memory pool
        alloc = cl.tools.ImmediateAllocator(self.__queue)
        self.__mem_pool = cl.tools.MemoryPool(alloc)
        # create tensor type for device
        name = "%s(%s)" % (OpenCLTensor.__name__, self.__desc)
        self.__tensor_cls = type(name, (OpenCLTensor,), {'_OpenCLTensor__device': self})

    @classmethod
    def from_context(cls, context:cl.Context) -> "OpenCLDevice":
        return cls.device_pool.get_device_from_cl_device(context.devices[0])

    @property
    def context(self) -> cl.Context:
        return self.__context
    @property
    def queue(self) -> cl.CommandQueue:
        return self.__queue
    @property
    def mem_pool(self) -> cl.tools.MemoryPool:
        return self.__mem_pool
    @property
    def Tensor(self) -> type:
        return self.__tensor_cls

    def __repr__(self):
        return str(self.__cl_device)
    def __str__(self):
        return f"{self.__class__.__name__}({self.__desc})"

    @classmethod
    def default_device(cls) -> "OpenCLDevice":
        if cls.__default_device is None:
            cls.__default_device = OpenCLDevice()
        return cls.__default_device

from .tensor import OpenCLTensor
