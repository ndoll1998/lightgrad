import numpy as np
import pyopencl as cl
from ..func import Function
from .tensor import OpenCLTensor
import functools

""" Helpers """

DTYPE2CTYPE = {
    # int
    'int16': 'short',
    'int32': 'int',
    'int64': 'long',
    # float
    'float32': 'float',
    'float64': 'double'
}

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

def atom_kernel(operation_str:str, **named_tensors):
    # TODO: handle non tensor inputs
    # populate inputs
    names, tensors, datas, ctypes = [], [], [], []
    for name, t in named_tensors.items():
        names.append(name)
        tensors.append(t)
        datas.append(t.data)
        ctypes.append(DTYPE2CTYPE[t.dtype.name])
    # TODO: broadcast output dtype
    assert all((ctype == ctypes[0] for ctype in ctypes[1:]))
    out_ctype = ctypes[0]
    # get device to use - all tensors are of the same type and therfore also on the same device 
    device = tensors[0].device

    if all((t.shape == tensors[0].shape for t in tensors[1:])):
        # no shape broadcasting
        prg = cache_build_kernel(device.ctx, """
            __kernel void fn(""" + \
                ' '.join(["const __global %s* %s," % (c, n.upper()) for n, c in zip(names, ctypes)]) + \
                """
                __global """ + out_ctype + """* Out
            ) {
                const uint i = get_global_id(0);
                """ + \
                '\n\t\t'.join(["const %s %s = %s[i];" % (c, n.lower(), n.upper()) for n, c in zip(names, ctypes)]) + \
                """           
                Out[i] = """ + operation_str + """;
            }
        """)
        # execute
        out = OpenCLTensor.empty(tensors[0].shape, dtype=tensors[0].dtype, device=device)
        prg.fn(device.queue, [out.numel()], None, *datas, out.data)
        return out

    else:
        # shape broadcast
        prg = cache_build_kernel(device.ctx, """
            __kernel void fn(
                // input tensors\n\t\t""" + \
                # const __global float* A,
                ' '.join(["const __global %s* %s," % (c, n.upper()) for n, c in zip(names, ctypes)]) + "\n\t\t" + \
                # const __global uint* A_shape,
                ' '.join(["const __global uint* %s_shape," % n.upper() for n in names]) + "\n\t\t" + \
                """ // output tensor
                __global """ + out_ctype + """* Out,
                const __global uint* Out_shape,
                // others
                const uint dims
            ) {
                uint size = get_global_size(0);
                const uint i = get_global_id(0);
                uint """ + \
                # uint idx_A = 0, idx_B = 0;
                ', '.join(['%s_idx = 0' % n.upper() for n in names]) + ";" +\
                """
                // get broadcasted flat index
                for (int d=0; d < dims; d++) {
                    size /= Out_shape[d];
                    const uint Out_idx = (i / size) % Out_shape[d];
                    """ + \
                    # idx_A = (idx_A * A_shape[d]) + (C_idx % A_shape[d]);
                    ('\n\t\t' + " " * 4).join(["{0}_idx = ({0}_idx * {0}_shape[d]) + (Out_idx % {0}_shape[d]);".format(n.upper()) for n in names]) + \
                    """
                }
                // gather elements by index and apply function
                """ + \
                # const float a = A[A_idx];
                '\n\t\t'.join(["const %s %s = %s[%s_idx];" % (c, n, n.upper(), n.upper()) for c, n in zip(ctypes, names)]) + \
                """
                Out[i] = """ + operation_str + """;
            }
        """)
        # broadcast shapes
        dims = max([len(t.shape) for t in tensors])
        shapes = []
        for t in tensors:
            shape = np.ones(dims, dtype=np.int32)
            shape[-len(t.shape):] = t.shape
            shapes.append(shape)
        # create output tensor
        out_shape = np.maximum(*shapes)
        assert all(np.all((s == 1) | (s == out_shape)) for s in shapes), "Shapes do not match!"
        out = OpenCLTensor.empty(out_shape, dtype=tensors[0].dtype, device=device)
        # create shape tensors and get their buffers
        shapes = [OpenCLTensor.from_numpy(s, requires_grad=False, device=device).data for s in shapes]
        out_shape = OpenCLTensor.from_numpy(out_shape, requires_grad=False, device=device).data
        # apply function
        prg.fn(device.queue, [out.numel()], None, *datas, *shapes, out.data, out_shape, np.int32(dims))
        return out


""" Basic Math Operators """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        return atom_kernel(
            a=a,
            operation_str="-a"
        )
    def backward(ctx, out_grad):
        return -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__add__")
@OpenCLTensor.register_op("__radd__")
class add(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b,
            operation_str="a + b"
        )
        # return
        return c
    def backward(ctx, out_grad):
        return out_grad, out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__sub__")
class sub(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b,
            operation_str="a - b"
        )
        # return
        return c
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__mul__")
@OpenCLTensor.register_op("__rmul__")
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b,
            operation_str="a * b"
        )
        # return
        return c
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad * b, a * out_grad
