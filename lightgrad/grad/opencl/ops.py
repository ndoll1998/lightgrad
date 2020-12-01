import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from ..func import Function
from .tensor import OpenCLTensor
import functools

""" Helpers """

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

@functools.lru_cache()
def cache_build_atom_kernel(ctx, operation_str:str, names:tuple, ctypes:tuple, out_tensor_id:int):
    assert len(names) == len(ctypes)
    # build arguments
    tensor_args = ["__global %(type)s* %(name)s" % {'type': c, 'name': n.upper()} for n, c in zip(names, ctypes)]    
    shape_args = ["__global uint* %(name)s_shape" % {'name': n.upper()} for n in names]
    stride_args = ["__global uint* %(name)s_strides" % {'name': n.upper()} for n in names]
    dimension_args = ["uint %(name)s_dim" % {'name': n.upper()} for n in names]
    # index initialize
    idx_names = ["%(name)s_idx" % {'name': n.upper()} for n in names]
    idx_init = "uint " + ', '.join(["%(idx_name)s = 0" % {'idx_name': n} for n in idx_names]) + ";"

    nl = '\n'
    update_idx = lambda n, idx: f"""
                diff = {names[out_tensor_id].upper()}_dim - {n.upper()}_dim;
                if ((d >= diff) && ({n.upper()}_shape[d - diff] > 1))
                    {idx} += k * {n.upper()}_strides[d - diff];
    """
    # build program source
    source = f"""
        __kernel void atom(
            {', '.join(tensor_args)},
            {', '.join(shape_args)},
            {', '.join(stride_args)},
            {', '.join(dimension_args)}
        ) {{
            // get worker information
            uint size = get_global_size(0);
            const uint i = get_global_id(0);            
            // initialize array indices
            {idx_init}
            // get indices for tensors
            for (int d=0; d < {names[out_tensor_id].upper()}_dim; ++d) {{
                uint out_d = {names[out_tensor_id].upper()}_shape[d];
                size /= out_d;

                //const uint j = i / size;
                //const uint l = j / out_d;   // this is what makes modulo slow
                //const uint k = j - l * out_d;
                const uint k = (i / size) % out_d;  // modulo is slow when influencing data accesses

                // update array indices
                uint diff; \
                {''.join([update_idx(n, idx) for n, idx in zip(names, idx_names)])}
            }}
            // gather elements by indices
            {
                (nl+" "*12).join([
                    f"const {t} {n} = {n.upper()}[{idx}];"
                    for t, n, idx in zip(
                        ctypes[:out_tensor_id] + ctypes[out_tensor_id+1:],
                        names[:out_tensor_id] + names[out_tensor_id+1:], 
                        idx_names[:out_tensor_id] + idx_names[out_tensor_id+1:]
                    )
                ])
            }
            // apply function
            {ctypes[out_tensor_id]} {names[out_tensor_id]};
            {operation_str};
            // save output in tensor
            {names[out_tensor_id].upper()}[{idx_names[out_tensor_id]}] = {names[out_tensor_id]};
        }}
    """
    return cl.Program(ctx, source).build()

def atom_kernel(operation_str:str, out:str ="__OUT", **named_tensors):
    # TODO: handle non tensor inputs
    names = tuple(sorted(named_tensors))       # make sure the order is always the same to reduce compilations
    tensors = tuple(named_tensors[n] for n in names)
    ctypes = tuple(dtype_to_ctype(t.dtype) for t in tensors)
    device = tensors[0].device
    if out not in names:
        # create output tensor 
        # TODO: broadcast dtype
        out_shape = np.maximum(*(t.shape for t in tensors))
        out_tensor = device.Tensor.empty(out_shape, dtype=tensors[0].dtype)
        # add output tensor
        names += (out,)
        tensors += (out_tensor,)
        ctypes += (dtype_to_ctype(out_tensor.dtype),)
    else:
        out_tensor = tensors[names.index(out)]
    # broadcast shape checking
    for t in tensors:
        assert len(t.shape) <= len(out_tensor.shape)
        assert np.all((np.asarray(t.shape) == 1) | (np.asarray(t.shape) == out_tensor.shape[-len(t.shape):]))
    # collect all kernel inputs
    datas = tuple(t.data for t in tensors)
    shapes = tuple(t._shape_buf for t in tensors)
    strides = tuple(t._strides_buf for t in tensors)
    dims = tuple(np.int32(len(t.shape)) for t in tensors)
    # build program and apply
    prg = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out))
    prg.atom(device.queue, [out_tensor.numel()], None, *datas, *shapes, *strides, *dims)
    device.queue.finish()
    # return
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
    
@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__pow__")
class pow(Function):
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return atom_kernel(
            a=a, b=b, out="o",
            operation_str="o = pow(a, b)"
        )
    

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
