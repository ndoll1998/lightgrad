import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from .tensor import OpenCLTensor
import functools
from math import ceil

@functools.lru_cache()
def cache_build_kernel(ctx, source):
    return cl.Program(ctx, source).build()

def _stride_idx_source(*names, 
    dim="dim",
    flat_id="get_global_id(0)", 
    flat_size="get_global_size(0)",
    get_cur_dim=lambda d: f"shape[{d}]",
    get_cur_stride=lambda name, i, d: f"strides[{i} * dim + {d}]"
):
    nl, idxs = '\n', ["%(name)s_idx" % {'name': n.upper()} for n in names]
    # strided index calculation source code
    return f"""{"uint " + ', '.join(["%(idx_name)s = 0" % {'idx_name': n} for n in idxs]) + ";"}
            {{
                uint contiguous_stride = {flat_size};
                // get indices for tensors
                for (int d=0; d < dim; ++d) {{
                    const uint s = {get_cur_dim('d')};
                    contiguous_stride /= s;
                    const uint k = ({flat_id} / contiguous_stride) % s;
                    // update array indices
                    {(nl + " "*20).join([
                        f"{idx} += k * {get_cur_stride(names[i], i, 'd')};"
                        for i, idx in enumerate(idxs)
                    ])}
                }}
            }}"""

@functools.lru_cache()
def cache_build_atom_kernel(ctx, operation_str:str, names:tuple, ctypes:tuple, out_tensor_id:int, use_strides:bool, read_out:bool):
    assert len(names) == len(ctypes)
    # build arguments
    tensor_args = [f"__global {ctype}* {name.upper()}" for name, ctype in zip(names, ctypes)]
    offset_args = [f"const uint {name.upper()}_off" for name in names]
    stride_args = ["const __global uint* strides", "const __global uint* shape", "const uint dim"] if use_strides else []
    # buffer indices with offsets
    idxs = [f"{name.upper()}_idx + {name.upper()}_off" for name in names] if use_strides else [f"i + {n.upper()}_off" for n in names]
    # build program source
    nl = "\n"
    source = f"""
        __kernel void atom(
            {', '.join(tensor_args + offset_args + stride_args)}
        ) {{
            // get worker information
            const uint i = get_global_id(0);
            // build strided indices if we need them
            {_stride_idx_source(*names, flat_id="i") if use_strides else ""}
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
            {ctypes[out_tensor_id]} {names[out_tensor_id]}{
                " = %s[%s]" % (names[out_tensor_id].upper(), idxs[out_tensor_id]) if read_out else ""};
            {operation_str};
            // save output in tensor
            {names[out_tensor_id].upper()}[{idxs[out_tensor_id]}] = {names[out_tensor_id]};
        }}
    """
    return cl.Program(ctx, source).build().atom

def atom_kernel(operation_str:str, out:str ="__OUT", depends_on_out:bool =True, **named_tensors):
    # get device to use
    t0 = next((t for t in named_tensors.values() if isinstance(t, OpenCLTensor)), None)
    assert t0 is not None, "No OpenCLTensor was provided!"
    device = t0.device
    # handle non tensor inputs
    for k, t in named_tensors.items():
        t = [t] if not isinstance(t, (np.ndarray, OpenCLTensor, tuple, list)) else t
        t = np.asarray(t, dtype=t0.dtype) if not isinstance(t, (np.ndarray, OpenCLTensor)) else t
        t = device.Tensor.from_numpy(t) if not isinstance(t, OpenCLTensor) else t
        named_tensors[k] = t
    # separate names and tensors
    names, tensors = zip(*named_tensors.items())
    names, tensors = tuple(names), tuple(tensors)
    # broadcast shape
    dim = max((len(t.shape)) for t in tensors)
    shapes = [(1,) * (dim - len(t.shape)) + t.shape for t in tensors]
    shapes = np.array(shapes, dtype=np.int32)
    shape = np.max(shapes, axis=0)
    assert (shapes[shapes != shape] == 1).all(), "Cannot broadcast shapes!"
    # build broadcasted strides
    strides = [(0,) * (dim - len(t.strides)) + t.strides for t in tensors]
    strides = np.array(strides, dtype=np.int32)
    strides[shapes != shape] = 0
    # create output tensor
    if out not in names:
        # TODO: output dtype depends on combination input dtypes
        out_dtype = t0.dtype
        out_tensor = device.Tensor.empty(shape, dtype=out_dtype)
        out_id = len(names)
        # add to tuples
        names += (out,)
        tensors += (out_tensor,)
        # add output strides
        strides = np.vstack((strides, out_tensor.strides))
        # cannot depend on output because output tensor is empty
        depends_on_out = False
    else:
        # get output id and tensor
        out_id = names.index(out)
        out_tensor = tensors[out_id]
        # make sure we can broadcast to output tensor
        assert (out_tensor.shape == shape).all, "Output tensor shape does not match broadcast shape! (%s != %s)" % (out_tensor.shape, tuple(shape))
    # collect data buffers and ctypes
    datas = tuple(t.data for t in tensors)
    offsets = tuple(np.int32(t.offset) for t in tensors)
    ctypes = tuple(dtype_to_ctype(t.dtype) for t in tensors)
    # check if strides all strides are equal
    if np.logical_and.reduce(strides[0, :] == strides[1:, :], axis=0).all():
        # kernel does not need to consider strides in this case
        knl = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=False, read_out=depends_on_out)
        knl.set_args(*datas, *offsets)
    else:
        # build kernel input buffers
        shape = cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=shape)
        strides = cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=strides)
        # get kernel and set arguments
        knl = cache_build_atom_kernel(device.ctx, operation_str, names, ctypes, names.index(out), use_strides=True, read_out=depends_on_out)
        knl.set_args(*datas, *offsets, strides, shape, np.int32(dim))
    # enqueue and wait until finished
    cl.enqueue_nd_range_kernel(device.queue, knl, [out_tensor.numel()], None)
    device.queue.finish()
    # return output tensor
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
            uint A_off,
            uint B_off,
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

            // offsets
            A_off += bi * TS * K;
            B_off += bj * TS;
            
            for (int t = 0; t < K; t+=RTS) {{                
                // load tile into local memory
                for (int r = 0; r < TS; r+=RTS) {{
                    Asub[lj][r+li] = A[A_off + (r + li) * K + t + lj];
                    Bsub[lj][r+li] = B[B_off + (lj + t) * N + r + li];
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
    """).build().matmul

def _match_blocks(T, block_size):
    M, N = T.shape
    if (M % block_size != 0) or (N % block_size != 0):
        shape = (ceil(M / block_size) * block_size, ceil(N / block_size) * block_size)
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
    knl = cache_build_dot_kernel(device.ctx, ctype_A, ctype_B, ctype_O, block_size, work_per_thread)
    knl(device.queue, global_shape, local_shape, 
            A.contiguous().data, B.contiguous().data, O.data, 
            np.int32(A.offset), np.int32(B.offset),
            pad_M, pad_N, pad_K)
    device.queue.finish()
    # remove padding from output
    return (O if (M, N) == (pad_M, pad_N) else O[:M, :N])


@functools.lru_cache()
def cache_build_reduction_kernel(ctx, operation_str:str, ctype:str, neutral:str, use_strides:bool):
    # additional arguments list
    stride_args = ["__global uint* strides", "__global uint* shape", "uint dim"] if use_strides else []
    nl = '\n'
    # build kernel
    return cl.Program(ctx, f"""
        __kernel void reduce(
            uint n_red_items,
            __global {ctype}* T,
            __global {ctype}* O,
            __local {ctype}* loc_buf,
            const uint T_off{(',' + nl + ' ' * 12).join([""] + stride_args)}
        ) {{
            uint gi = get_global_id(0);
            uint gj = get_global_id(1);
            uint lj = get_local_id(1);

            uint group_id = get_group_id(0) * get_num_groups(1) + get_group_id(1);
            uint group_size = get_local_size(1);

            // compute strided index if necessary
            uint i = gi * n_red_items + gj;
            {
                _stride_idx_source("T", 
                    flat_id="i",
                    flat_size="get_global_size(0) * n_red_items",
                    get_cur_stride=lambda n, i, d: f"strides[{d}]"
                ) if use_strides else "uint T_idx = i;"
            }
            // load to local memory
            loc_buf[lj] = (gj < n_red_items)? T[T_off + T_idx] : {neutral};
            barrier(CLK_LOCAL_MEM_FENCE);
            // reduce
            for(uint s = group_size/2; s > 0; s >>= 1) {{
                if (lj < s) {{
                    {ctype} a = loc_buf[lj];
                    {ctype} b = loc_buf[lj+s];
                    loc_buf[lj] = {operation_str};
                }}
                barrier(CLK_LOCAL_MEM_FENCE);
            }}
            // store partial sums in output
            if(lj == 0) O[group_id] = loc_buf[0];
        }}
    """).build().reduce

def reduction_kernel(T, axis:int, keepdims:bool, operation_str:str, neutral:str ="0", group_size:int =128):
    # get device
    device = T.device
    # prepare axis
    axis = tuple(range(len(T.shape))) if axis is None else (axis,) if not isinstance(axis, tuple) else axis
    axis = tuple(i if i >= 0 else (len(T.shape) + i) for i in axis)
    # number of iterations needed for full reduction
    n_red_items = np.prod([T.shape[i] for i in axis])
    n_groups = ceil(n_red_items / group_size)
    n_iters = ceil(np.log(n_red_items) / np.log(group_size))
    # build output shape and create output tensor
    shape = tuple(s if i not in axis else 1 for i, s in enumerate(T.shape) if (i not in axis) or keepdims)
    shape = (1,) if len(shape) == 0 else shape
    shape += (n_groups,)
    O = device.Tensor.empty(shape, dtype=T.dtype)
    # transpose to have reduction dimensions at last
    if len(axis) < len(T.shape):
        perm = list(range(len(T.shape)))
        for i, j in enumerate(axis, 1):
            perm[-i], perm[j] = perm[j], perm[-i]
        T = T.transpose(*perm)
    # gather kernel relevant information
    ctype = dtype_to_ctype(T.dtype)
    use_strides = (len(axis) < len(T.shape)) and (not T.is_contiguous)

    # build kernels
    first_knl = cache_build_reduction_kernel(device.ctx, operation_str, ctype, neutral, use_strides=use_strides)
    further_knl = cache_build_reduction_kernel(device.ctx, operation_str, ctype, neutral, use_strides=False)
    # local memory
    loc_buf = cl.LocalMemory(T.dtype.itemsize * group_size)
    # stride arguments for first iteration
    stride_args = (
        cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(T.strides, dtype=np.int32)),
        cl.Buffer(device.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(T.shape, dtype=np.int32)),
        np.int32(len(T.shape))
    ) if use_strides else tuple()

    # set kernel arguments
    first_knl.set_args(np.int32(n_red_items), T.data, O.data, loc_buf, np.int32(T.offset), *stride_args)
    # first iteration of reduction
    n, m = O.numel() // n_groups, ceil(n_red_items / group_size) * group_size
    cl.enqueue_nd_range_kernel(device.queue, first_knl, [n, m], [1, group_size])
    # further reduction iterations
    for i in range(1, n_iters):
        k = ceil(n_red_items / (group_size ** i))
        m = ceil(k / group_size) * group_size
        further_knl.set_args(np.int32(k), O.data, O.data, loc_buf, np.int32(0))
        cl.enqueue_nd_range_kernel(device.queue, further_knl, [n, m], [1, group_size])
    # wait for queue to finish
    device.queue.finish()
    # remove partial sums stored in last dimension of output
    n = int(np.prod(shape[:-1])) * O.dtype.itemsize
    return device.Tensor(O.data, shape=shape[:-1], dtype=O.dtype)
