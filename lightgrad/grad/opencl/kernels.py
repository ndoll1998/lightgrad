import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from .tensor import OpenCLTensor
import functools
from math import ceil

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

@functools.lru_cache()
def cache_build_atom_kernel(ctx, operation_str:str, names:tuple, ctypes:tuple, out_tensor_id:int, use_strides:bool, read_out:bool):
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
                f"{t} {n} = {n.upper()}[{idx}];"
                for t, n, idx in zip(
                    ctypes[:out_tensor_id] + ctypes[out_tensor_id+1:],
                    names[:out_tensor_id] + names[out_tensor_id+1:], 
                    idxs[:out_tensor_id] + idxs[out_tensor_id+1:]
                )
            ])}
            // apply function
            {ctypes[out_tensor_id]} {names[out_tensor_id]} { "= %s[%s]" % (names[out_tensor_id].upper(), idxs[out_tensor_id]) if read_out else "" };
            {operation_str};
            // save output in tensor
            {names[out_tensor_id].upper()}[{idxs[out_tensor_id]}] = {names[out_tensor_id]};
        }}
    """
    return cl.Program(ctx, source).build()

def atom_kernel(operation_str:str, out:str ="__OUT", depends_on_out:bool =True, **named_tensors):
    # get device to use
    t = next((t for t in named_tensors.values() if isinstance(t, OpenCLTensor)), None)
    device, dtype = t.device, t.dtype
    assert device is not None, "Cannot find device to use because no OpenCLTensor was provided!"
    # handle non tensor inputs
    for key, t in named_tensors.items():
        t = np.asarray(t, dtype=dtype) if not isinstance(t, (np.ndarray, OpenCLTensor)) else t
        t = device.Tensor.from_numpy(t) if not isinstance(t, OpenCLTensor) else t
        named_tensors[key] = t
    # prepare
    names = tuple(sorted(named_tensors))       # make sure the order is always the same to reduce compilations
    tensors = tuple(named_tensors[n] for n in names)
    ctypes = tuple(dtype_to_ctype(t.dtype) for t in tensors)
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
        # cannot depend on output since it was just created
        depends_on_out = False
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
        prg = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=True, read_out=depends_on_out)
        prg.atom(device.queue, [out_tensor.numel()], None, *datas, *strides, shape, np.int32(dim))
    else:
        # build program and apply
        prg = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=False, read_out=depends_on_out)
        prg.atom(device.queue, [out_tensor.numel()], None, *datas)
    # wait until operation is finished
    device.queue.finish()
    return out_tensor


@functools.lru_cache()
def cache_build_dot_kernel(ctx, ctype_A:str, ctype_B:str, ctype_O:str, block_size:int, work_per_thread:int):
    return cl.Program(ctx, f"""
        #define TS {block_size}         // tile size
        #define WPT {work_per_thread}   // work per thread
        #define RTS TS/WPT              // reduced tile size

        __kernel void matmul(
            const __global {ctype_A}* A,
            const __global {ctype_B}* B,
            __global {ctype_O}* O,
            const uint M,
            const uint N,
            const uint K
        ) {{
            // local thread index
            const uint li = get_local_id(0);
            const uint lj = get_local_id(1);
            // work-group index
            const uint bi = get_group_id(0);
            const uint bj = get_group_id(1);

            // allocate local memory
            __local {ctype_A} Asub[RTS][TS];
            __local {ctype_B} Bsub[RTS][TS];
            // allocate private registers
            {ctype_A} Areg;
            {ctype_B} Breg[WPT];
            {ctype_O} acc[WPT][WPT];

            // zero out accumulation registers
            for (int wi=0; wi < WPT; wi++)
                for (int wj=0; wj < WPT; wj++)
                    acc[wi][wj] = 0;

            // read offsets
            const uint Aoff = bi * TS * K;
            const uint Boff = bj * TS;
            
            for (int t = 0; t < K; t+=RTS) {{                
                // load tile into local memory
                for (int r = 0; r < TS; r+=RTS) {{
                    Asub[lj][r+li] = A[Aoff + (r + li) * K + t + lj];
                    Bsub[lj][r+li] = B[Boff + (lj + t) * N + r + li];
                }}
                // wait until all loaded
                barrier(CLK_LOCAL_MEM_FENCE);

                for (int k = 0; k < RTS; k++) {{
                    // load values of Bsub to registers
                    for (int wj=0; wj < WPT; wj++) 
                        Breg[wj] = Bsub[k][lj + wj * RTS];
                    // accumulate
                    for (int wi=0; wi < WPT; wi++) {{
                        Areg = Asub[k][li + wi * RTS];
                        for (int wj=0; wj<WPT; wj++)
                            acc[wj][wi] += Areg * Breg[wj];
                    }}
                }}
                // wait until work group finished computations
                // before loading next tiles into local memory
                barrier(CLK_LOCAL_MEM_FENCE);
            }}
            // store output in matrix            
            uint i = bi * TS + li;
            uint j = bj * TS + lj;
            for (int wi = 0; wi < WPT; wi++)
                for (int wj = 0; wj < WPT; wj++)
                    O[(i + wj*RTS) * N + (j + wi*RTS)] = acc[wi][wj];
        }}
    """).build()

def _match_blocks(T, block_size):
    M, N = T.shape
    if (M % block_size != 0) or (N % block_size != 0):
        shape = (M + block_size - M % block_size, N + block_size - N % block_size)
        T_pad = T.device.Tensor.zeros(shape, dtype=T.dtype)
        T_pad[:M, :N] = T
        return T_pad
    return T

def dot_kernel(A, B, block_size:int =128, work_per_thread:int =8):
    assert len(A.shape) == len(B.shape) == 2
    assert A.shape[1] == B.shape[0]
    # get tensor information
    device = A.device
    M, N = np.int32(A.shape[0]), np.int32(B.shape[1])
    # pad inputs to be multiple of block size in both directions
    A = _match_blocks(A, block_size)
    B = _match_blocks(B, block_size)
    # create output tensor
    pad_M, pad_N, pad_K = np.int32(A.shape[0]), np.int32(B.shape[1]), np.int32(A.shape[1])
    O = device.Tensor.empty(shape=(pad_M, pad_N), dtype=A.dtype) # TODO: broadcast dtype
    # get data types
    ctype_A = dtype_to_ctype(A.dtype)
    ctype_B = dtype_to_ctype(B.dtype)
    ctype_O = dtype_to_ctype(O.dtype)
    # build and call kernel
    global_shape, local_shape = [pad_M // work_per_thread, pad_N // work_per_thread], [block_size // work_per_thread] * 2
    prg = cache_build_dot_kernel(device.ctx, ctype_A, ctype_B, ctype_O, block_size, work_per_thread)
    prg.matmul(device.queue, global_shape, local_shape, A.contiguous().data, B.contiguous().data, O.data, pad_M, pad_N, pad_K)
    # remove padding from output
    return (O if (M, N) == (pad_M, pad_N) else O[:M, :N])
