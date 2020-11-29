import pyopencl as cl
from ..func import Function
from .tensor import OpenCLTensor
from numpy.ctypeslib import as_ctypes_type as get_ctype
import functools

""" Helpers """

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

def unary_atom_kernel(ctx, dtype, operation_str:str):
    ctype = get_ctype(dtype).__name__[2:]
    return cache_build_kernel(ctx, """
        __kernel void fn(
            const __global """ + ctype + """* A,
            __global """ + ctype + """* B
        ) {
            const uint i = get_global_id(0);
            """ + operation_str + """;
        }
    """)

def binary_atom_kernel(ctx, a_dtype, b_dtype, c_dtype, operation_str:str):
    a_ctype = get_ctype(a_dtype).__name__[2:]
    b_ctype = get_ctype(b_dtype).__name__[2:]
    c_ctype = get_ctype(c_dtype).__name__[2:]
    return cache_build_kernel(ctx, """
        __kernel void fn(
            const __global """ + a_ctype + """* A,
            const __global """ + b_ctype + """* B,
            __global """ + c_ctype + """* C
        ) {
            const uint i = get_global_id(0);
            """ + operation_str + """;
        }
    """)


""" Basic Math Operators """

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__neg__")
class neg(Function):
    def forward(ctx, a):
        # build program
        prg = unary_atom_kernel(
            ctx=a.device.ctx,
            dtype=a.dtype,
            operation_str="B[i] = -A[i]"
        )
        # execute
        b = OpenCLTensor.empty(a.shape, dtype=a.dtype, device=a.device)
        prg.fn(a.device.queue, [a.numel()], None, a.data, b.data)
        # return
        return b
    def backward(ctx, out_grad):
        return -out_grad

@OpenCLTensor.register_op()
@OpenCLTensor.register_op("__add__")
class add(Function):
    def forward(ctx, a, b):
        # TODO: broadcast shape and dtype
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        # build program
        prg = binary_atom_kernel(
            ctx=a.device.ctx,
            a_dtype=a.dtype,
            b_dtype=b.dtype,
            c_dtype=a.dtype,
            operation_str="C[i] = A[i] + B[i]"
        )
        # execute
        c = OpenCLTensor.empty(a.shape, dtype=a.dtype, device=a.device)
        prg.fn(a.device.queue, [a.numel()], None, a.data, b.data, c.data)
        # return
        return c
    def backward(ctx, out_grad):
        return out_grad, out_grad
