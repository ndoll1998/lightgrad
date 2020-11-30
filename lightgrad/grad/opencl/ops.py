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

non_broadcast_atom_source = lambda operation_str, names, ctypes, out_tensor_id: """
    __kernel void fn(\n\t""" + \
        ', '.join(["__global %s* %s" % (c, n.upper()) for n, c in zip(names, ctypes)]) + \
        """
    ) {
        const uint i = get_global_id(0);
        """ + \
        '\n\t'.join(["const %s %s = %s[i];" % (c, n.lower(), n.upper()) 
            for i, (n, c) in enumerate(zip(names, ctypes)) if i != out_tensor_id]) + \
        """           
        """ + operation_str + """;
    }
"""

broadcast_atom_source = lambda operation_str, names, ctypes, out_tensor_id: """
    __kernel void fn(
        // tensors\n\t""" + \
        ' '.join(["__global %s* %s," % (c, n.upper()) for n, c in zip(names, ctypes)]) + "\n\t" + \
        ' '.join(["const __global uint* %s_shape," % n.upper() for n in names]) + "\n\t" + \
        """ // others
        const uint dims
    ) {
        uint size = get_global_size(0);
        const uint i = get_global_id(0);
        uint """ + \
        ', '.join(['%s_idx = 0' % n.upper() for i, n in enumerate(names) if i != out_tensor_id]) + ";" +\
        """
        // get broadcasted flat index
        for (int d=0; d < dims; d++) {
            """ + "size /= %s_shape[d];" % names[out_tensor_id].upper() + """
            """ + "const uint {0}_idx = (i / size) % {0}_shape[d];".format(names[out_tensor_id].upper()) + """
            """ + \
            ('\n\t' + " "*4).join(["{0}_idx = ({0}_idx * {0}_shape[d]) + ({1}_idx % {0}_shape[d]);".format(
                n.upper(), names[out_tensor_id].upper()) for i, n in enumerate(names) if i != out_tensor_id]) + \
            """
        }
        // gather elements by index and apply function
        """ + \
        '\n\t'.join(["const %s %s = %s[%s_idx];" % (c, n.lower(), n.upper(), n.upper()) 
            for i, (c, n) in enumerate(zip(ctypes, names)) if i != out_tensor_id]) + \
        """
        """ + operation_str + """;
    }
"""

def atom_kernel(operation_str:str, out:str ="__OUT", **named_tensors):
    # TODO: handle non tensor inputs
    # populate inputs
    names, tensors, datas, ctypes = [], [], [], []
    for name, t in named_tensors.items():
        names.append(name)
        tensors.append(t)
        datas.append(t.data)
        ctypes.append(DTYPE2CTYPE[t.dtype.name])
    # is inplace if output is one of the inputs
    is_inplace = (out in names)
    # TODO: broadcast output dtype
    assert all((ctype == ctypes[0] for ctype in ctypes[1:]))
    out_dtype = tensors[0].dtype
    out_ctype = ctypes[0]
    # get device to use - all tensors are of the same type and therfore also on the same device 
    device = tensors[0].device

    # do tensors need broadcasting
    broadcast = any((t.shape != tensors[0].shape for t in tensors[1:]))
    # create output tensor
    if broadcast:
        # create broadcasted shape
        dims = max([len(t.shape) for t in tensors])
        shapes = []
        for t in tensors:
            shape = np.ones(dims, dtype=np.int32)
            shape[-len(t.shape):] = t.shape
            shapes.append(shape)
        # create output shape
        out_shape = np.maximum(*shapes)
        assert all(np.all((s == 1) | (s == out_shape)) for s in shapes), "Shapes do not match!"
        # only add to shapes if new output tensor will be created
        if not is_inplace:
            shapes.append(out_shape)
    else:
        out_shape = tensors[0].shape

    if not is_inplace:
        # create output tensor
        out_tensor_id = len(names)
        out_tensor = OpenCLTensor.empty(out_shape, dtype=out_dtype, device=device)
        # add output tensor to populated inputs
        names.append(out)
        tensors.append(out_tensor)
        datas.append(out_tensor.data)
        ctypes.append(out_ctype)

    # get tensor to manipulate output
    out_tensor_id = names.index(out)
    out_tensor = tensors[out_tensor_id]

    # build and execute program
    if not broadcast:
        # no shape broadcasting
        source = non_broadcast_atom_source(operation_str, names, ctypes, out_tensor_id)
        prg = cache_build_kernel(device.ctx, source)
        # execute
        prg.fn(device.queue, [out_tensor.numel()], None, *datas)
    else:
        # broadcast
        source = broadcast_atom_source(operation_str, names, ctypes, out_tensor_id)
        prg = cache_build_kernel(device.ctx, source)
        # create shape tensors and apply function
        shapes = [OpenCLTensor.from_numpy(s, requires_grad=False, device=device).data for s in shapes]
        prg.fn(device.queue, [out_tensor.numel()], None, *datas, *shapes, np.int32(dims))
    return out_tensor
    
""" Transformations """

@OpenCLTensor.register_op()
class reshape(Function):
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        return OpenCLTensor(a.data, shape=shape, dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(shape)

""" Basic Math Operators """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        return atom_kernel(
            a=a, out="o",
            operation_str="O[i] = -a"
        )
    def backward(ctx, out_grad):
        return -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__add__")
@OpenCLTensor.register_op("__radd__")
class add(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="O[i] = a + b"
        )
    def backward(ctx, out_grad):
        return out_grad, out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__sub__")
class sub(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="O[i] = a - b"
        )
    def backward(ctx, out_grad):
        return out_grad, -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__mul__")
@OpenCLTensor.register_op("__rmul__")
class mul(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="O[i] = a * b"
        )
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        return out_grad * b, a * out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__div__")
class div(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="O[i] = a / b"
        )
    
@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__pow__")
class pow(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="O[i] = pow(a, b)"
        )
    

""" Inplace Operators """

@OpenCLTensor.register_op('__iadd__')
class __iadd(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="A[i] += b"
        )

@OpenCLTensor.register_op('__isub__')
class __isub(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="A[i] -= b"
        )

@OpenCLTensor.register_op('__imul__')
class __imul(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="A[i] *= b"
        )
        
@OpenCLTensor.register_op('__itruediv__')
class __idiv(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="A[i] /= b"
        )

@OpenCLTensor.register_op()
class fill(Function):
    def forward(ctx, t, val):
        val = np.asarray(val, dtype=t.dtype)
        cl.enqueue_fill_buffer(t.device.queue, t.data, val, 0, t.data.size)
        return t
