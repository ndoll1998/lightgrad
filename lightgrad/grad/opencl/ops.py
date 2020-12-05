import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from ..func import Function
from ..grads import Gradients
from .tensor import OpenCLTensor
import functools

""" Helpers """

def _bi_reverse(f):
    """ reverse inputs of bi-operator """
    class F(f):
        def forward(ctx, a, b):
            return f.forward(ctx, b, a)
        def backward(ctx, out_grad):
            return reversed(f.backward(out_grad))
    return F

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

@functools.lru_cache()
def cache_build_atom_kernel(ctx, operation_str:str, names:tuple, ctypes:tuple, out_tensor_id:int, use_strides:bool):
    assert len(names) == len(ctypes)
    # build arguments
    tensor_args = ["__global %(type)s* %(name)s" % {'type': c, 'name': n.upper()} for n, c in zip(names, ctypes)]    

    nl = '\n'
    if use_strides:
        # build kernel arguments
        stride_args = ["__global uint* %(name)s_strides" % {'name': n.upper()} for n in names]
        stride_args = f""",
            {', '.join(stride_args)},
            __global const uint* shape,
            const uint dim
        """
        # index initialize
        idxs = ["%(name)s_idx" % {'name': n.upper()} for n in names]
        idx_init = "uint " + ', '.join(["%(idx_name)s = 0" % {'idx_name': n} for n in idxs]) + ";"
        # strided index calculation source code
        build_indices_source = f"""{idx_init}
            uint size = get_global_size(0);
            // get indices for tensors
            for (int d=0; d < dim; ++d) {{
                size /= shape[d];
                uint j = (i / size) % shape[d];
                // update array indices
                {(nl + " "*16).join([
                    f"{idx} += j * {n.upper()}_strides[d];"
                    for n, idx in zip(names, idxs)
                ])}
            }}
        """
    else:
        # use global id as a index for all tensors
        idxs = "i"*len(names)

    # build program source
    source = f"""
        __kernel void atom(
            {', '.join(tensor_args)}{stride_args if use_strides else ""}
        ) {{
            // get worker information
            const uint i = get_global_id(0);
            // build strided indices if we need them
            {build_indices_source if use_strides else ""}
            // gather elements by indices
            {(nl+" "*12).join([
                f"const {t} {n} = {n.upper()}[{idx}];"
                for t, n, idx in zip(
                    ctypes[:out_tensor_id] + ctypes[out_tensor_id+1:],
                    names[:out_tensor_id] + names[out_tensor_id+1:], 
                    idxs[:out_tensor_id] + idxs[out_tensor_id+1:]
                )
            ])}
            // apply function
            {ctypes[out_tensor_id]} {names[out_tensor_id]};
            {operation_str};
            // save output in tensor
            {names[out_tensor_id].upper()}[{idxs[out_tensor_id]}] = {names[out_tensor_id]};
        }}
    """
    return cl.Program(ctx, source).build()

def atom_kernel(operation_str:str, out:str ="__OUT", **named_tensors):
    # TODO: handle non tensor inputs
    names = tuple(sorted(named_tensors))       # make sure the order is always the same to reduce compilations
    tensors = tuple(named_tensors[n] for n in names)
    ctypes = tuple(dtype_to_ctype(t.dtype) for t in tensors)
    device = tensors[0].device
    # broadcast shapes
    dim = max([len(t.shape) for t in tensors])
    shapes = []
    for t in tensors:
        shape = np.ones(dim, dtype=np.int32)
        shape[-len(t.shape):] = t.shape
        shapes.append(shape)
    shape = np.maximum(*shapes) if len(shapes) > 1 else shapes[0]
    assert all([np.all((s == 1) | (s == shape)) for s in shapes]), "Cannot broadcast shapes!"
    # create output tensor
    if out not in names:
        # TODO: broadcast dtype
        out_dtype = tensors[0].dtype
        out_tensor = device.Tensor.empty(shape, dtype=out_dtype)
        # add output tensor
        names += (out,)
        tensors += (out_tensor,)
        shapes += (shape,)
        ctypes += (dtype_to_ctype(out_dtype),)
    else:
        out_tensor = tensors[names.index(out)]
    # broadcast strides
    all_strides = []
    for t, s in zip(tensors, shapes):
        k = len(t.shape)
        strides = np.zeros(dim, dtype=np.int32)
        mask = (s[-k:] == shape[-k:])
        strides[-k:][mask] = np.asarray(t.strides, dtype=np.int32)[mask]
        all_strides.append(strides)
    # collect data buffers
    datas = tuple(t.data for t in tensors)
    # check if strides are necessary
    if any((st1 != st2).any() for st1, st2 in zip(all_strides[1:], all_strides[:-1])):
        # build kernel input buffers
        shape = cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=shape)
        strides = tuple(cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=st) for st in all_strides)
        # build program and apply
        prg = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=True)
        prg.atom(device.queue, [out_tensor.numel()], None, *datas, *strides, shape, np.int32(dim))
    else:
        # build program and apply
        prg = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=False)
        prg.atom(device.queue, [out_tensor.numel()], None, *datas)
    # wait until operation is finished
    device.queue.finish()
    return out_tensor

""" Transformations """

@OpenCLTensor.register_op()
class transpose(Function):
    def forward(ctx, a, *axes):
        assert len(axes) == len(a.shape)
        ctx.save_for_backward(axes)
        shape = tuple(a.shape[i] for i in axes)
        strides = tuple(a.strides[i] for i in axes)
        return OpenCLTensor(a.data, shape=shape, strides=strides, dtype=a.dtype)
    def backward(ctx, out_grad):
        axes, = ctx.get_saved_tensors()
        rev_axes = [None] * len(axes)
        for i, j in enumerate(axes):
            rev_axes[j] = i
        return out_grad.transpose(*rev_axes)

@OpenCLTensor.register_op()
class contiguous(Function):
    def forward(ctx, a):
        if not a.is_contiguous:
            return atom_kernel(
                a=a, out='o',
                operation_str='o = a'
            )
        return a
    def backward(ctx, out_grad):
        return out_grad

@OpenCLTensor.register_op()
class reshape(Function):
    @Gradients.no_grad()
    def forward(ctx, a, *shape):
        ctx.save_for_backward(a.shape)
        return OpenCLTensor(a.contiguous().data, shape=shape, dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, = ctx.get_saved_tensors()
        return out_grad.reshape(*shape)

""" Basic Math Operators """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        return atom_kernel(
            a=a, out="o",
            operation_str="o = -a"
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
            operation_str="o = a + b"
        )
    def backward(ctx, out_grad):
        return out_grad, out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__sub__")
class sub(Function):
    def forward(ctx, a, b):
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = a - b"
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
            operation_str="o = a * b"
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
            operation_str="o = a / b"
        )
    def backward(ctx, out_grad):
        a, b = ctx.get_saved_tensors()
        a_grad = atom_kernel(g=out_grad, b=b, out='o', operation_str='o = g / b')
        b_grad = atom_kernel(g=out_grad, a=a, b=b, out='o', operation_str='o = -a / pow(b, 2) * g')
        return a_grad, b_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__pow__")
class pow(Function):
    def forward(ctx, a, b):
        y = atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = pow(a, b)"
        )
        ctx.save_for_backward(a, b, y)
        return y
    def backward(ctx, out_grad):
        a, b, y = ctx.get_saved_tensors()
        a_grad = atom_kernel(g=out_grad, a=a, b=b, out='o', operation_str='o = b * pow(a, b-1) * g')
        b_grad = atom_kernel(g=out_grad, a=a, y=y, out='o', operation_str='o = g * y * log(a)')
        return a_grad, b_grad

# reverse operators for non-symmetrical operators
OpenCLTensor.register_op("__rsub__", _bi_reverse(sub))
OpenCLTensor.register_op("__rtruediv__", _bi_reverse(div))
OpenCLTensor.register_op("__rpow__", _bi_reverse(pow))
# OpenCLTensor.register_op("__rmatmul__", _bi_reverse(dot))

""" Inplace Operators """

@OpenCLTensor.register_op('__iadd__')
class __iadd(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a += b"
        )

@OpenCLTensor.register_op('__isub__')
class __isub(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a -= b"
        )

@OpenCLTensor.register_op('__imul__')
class __imul(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a *= b"
        )
        
@OpenCLTensor.register_op('__itruediv__')
class __idiv(Function):
    def forward(ctx, t, other):
        return atom_kernel(
            a=t, b=other, out='a',
            operation_str="a /= b"
        )

@OpenCLTensor.register_op()
class fill(Function):
    def forward(ctx, t, val):
        val = np.asarray(val, dtype=t.dtype)
        cl.enqueue_fill_buffer(t.device.queue, t.data, val, 0, t.data.size)
        return t

""" Non-Linearities """

""" Selectors """

@OpenCLTensor.register_op("__getitem__")
class __getitem(Function):
    """ TODO: currently only supports direct item indexing (no slicing or masking). """
    def forward(ctx, a, idx):
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(a.shape)
        idx = np.asarray(idx, dtype=np.int32)
        ctx.save_for_backward(a.shape, idx)
        off = (idx * a.strides).sum() * a.dtype.itemsize
        return OpenCLTensor(a.data[off:off+a.dtype.itemsize], shape=tuple(), dtype=a.dtype)
    def backward(ctx, out_grad):
        shape, idx = ctx.get_saved_tensors()
        grad = OpenCLTensor.zeros(shape, requires_grad=False, device=out_grad.device)
        cl.enqueue_copy(grad.device.queue, grad.data, out_grad.data, 
            byte_count=grad.dtype.itemsize,
            dest_offset=(idx * grad.strides).sum() * grad.dtype.itemsize
        )
        return grad

@OpenCLTensor.register_op("__setitem__")
class __setitem(Function):
    """ TODO: currently only supports direct item indexing (no slicing or masking). """
    def forward(ctx, a, idx, val):
        val = val if val is isinstance(val, np.ndarray) else np.asarray(val, dtype=a.dtype)
        idx = (idx,) if isinstance(idx, int) else idx
        assert len(idx) == len(a.shape)
        idx = np.asarray(idx, dtype=np.int32)
        cl.enqueue_copy(a.device.queue, a.data, val, 
            device_offset=(idx * a.strides).sum() * a.dtype.itemsize
        )
        return a