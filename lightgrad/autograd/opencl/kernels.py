
import numpy as np
import pyopencl as cl
from .tensor import OpenCLTensor
# tools
from functools import lru_cache, reduce
from itertools import zip_longest, chain
# utils
from pyopencl.tools import dtype_to_ctype
from numpy import uint32 as i32
from math import ceil, log2
# typing
from typing import Tuple, Union

__all__ = ['atom', 'dot', 'reduction']

#
#   Elementwise Kernel
#

@lru_cache()
def build_atom_kernel(context:cl.Context, 
    op:str,                 # operation to execute on variables
    # buffers
    buffers:tuple,          # unique names of variables / buffers
    buffer_dtypes:tuple,    # types of variables / buffers
    ndim:int,               # number of dimensions
    # scalars
    scalars:tuple,          # unique names of scalar inputs
    scalar_dtypes,          # types of scalars
    # read/write info
    read:tuple,             # buffers to read input from
    write:tuple             # buffers to write output to - must be a subset of buffers
) -> cl.Kernel:
    # assertions
    assert len(buffers) == len(buffer_dtypes),    "Variable names and data types do not align!"
    assert len(scalars) == len(scalar_dtypes),    "Variable names and data types do not align!"
    assert len(set(buffers)) == len(buffers),     "Variable names must be unique!"
    assert all(n in buffers for n in read),       "Reads must be contained in variable names!"
    assert all(n in buffers for n in write),      "Writes must be contained in variable names!"
    # prepare scalars
    scalars = tuple(n.lower() for n in scalars)
    scalar_ctypes = tuple(dtype_to_ctype(t) for t in scalar_dtypes)
    # prepare buffers
    values = tuple(n.lower() for n in buffers)
    buffers = tuple(n.upper() for n in buffers)
    buffer_ctypes = tuple(dtype_to_ctype(t) for t in buffer_dtypes)
    # prepare reads
    read = set(read)    # make sure to only read once
    read_ctypes = tuple(buffer_ctypes[buffers.index(n.upper())] for n in read)
    read_values = tuple(n.lower() for n in read)
    read_buffers = tuple(n.upper() for n in read)
    # prepare writes
    write = set(write)  # make sure to only write once
    write_ctypes = tuple(buffer_ctypes[buffers.index(n.upper())] for n in write)
    write_values = tuple(n.lower() for n in write)
    write_buffers = tuple(n.upper() for n in write)
    # create kernel source
    nl = lambda i=0: ('\n' + ' ' * i)
    source = f"""
        __kernel void atom(
            // buffers
            {' '.join(["__global %s* %s_data," % (c, N) for N, c in zip(buffers, buffer_ctypes)])}
            // scalars
            {' '.join(["%s %s," % (c, n) for c, n in zip(scalar_ctypes, scalars)])}
            // shape
            {', '.join(["const int size_%i" % i for i in range(ndim)])},
            // strides
            {(',' + nl(12)).join([
                ', '.join(["const int stride_%s_%i" % (N, i) for i in range(ndim)])
                for N in buffers
            ])},
            // offsets
            {', '.join(["const int offset_%s" % N for N in buffers])},
            // number of elements to compute
            const int N
        ) {{
            // gather work-item information
            const int i = get_global_id(0);
            const int n = get_global_size(0);

            for (int j = i; j < N; j+=n) {{
                // unflatten indices
                uint {", ".join(["%s_idx = offset_%s" % (N, N) for N in buffers])};
                {{
                    uint pos, j_ = j;
                    {nl(20).join([
                        ("pos = j_ %% size_%i;" % d) + nl(20) + 
                        nl(20).join(["%s_idx += pos * stride_%s_%i;" % (N, N, d) for N in buffers]) + 
                        ((nl(20) + "j_ /= size_%i;" % d) if d != 0 else "")
                        for d in range(ndim-1, -1, -1)
                    ])}
                }}
                // load data
                {nl(16).join(["%s %s = %s_data[%s_idx];" % (c, n, N, N) for c, n, N in zip(read_ctypes, read_values, read_buffers)])}
                // execute operation
                {' '.join(["%s %s;" % (c, n) for (c, n) in zip(write_ctypes, write_values) if n not in read_values])}
                {op};
                // store output
                {nl(12).join(["%s_data[%s_idx] = %s;" % (N, N, n) for N, n in zip(write_buffers, write_values)])}
            }}
        }}
    """
    # build program
    return cl.Program(context, source).build().atom

def _collapse_contiguous_dims(shape, strides):
    collapsed_shape = [shape[-1]]
    collapsed_strides = [[st[-1]] for st in strides] 
    for i in range(len(shape) - 2, -1, -1):
        if all(st[i+1] * shape[i+1] == st[i] for st in strides):
            # collapsable
            collapsed_shape[0] *= shape[i]
        else:
            # not collapsable
            collapsed_shape.insert(0, shape[i])
            # update strides
            for j, st in enumerate(strides):
                collapsed_strides[j].insert(0, st[i])
    return collapsed_shape, collapsed_strides

def atom(op:str, 
    # input / output tensors
    additional_read:tuple=tuple(),  # by default we only read the values of tensors mentioned in output
    output=('o',),                  # output tensors, if not mentioned in named tensors then a new tensor is created
    # kernel information
    block_size:int =256,            # local block size
    # inputs
    **named_tensors                 # all named tensors (except output tensors) needed for execution of op
) -> Tuple[OpenCLTensor]:
    # separate tensors from scalars
    named_scalars = {n: v for n, v in named_tensors.items() if not isinstance(v, OpenCLTensor)}
    named_tensors = {n: v for n, v in named_tensors.items() if isinstance(v, OpenCLTensor)}
    # separate names and values
    tensor_names, tensors = zip(*named_tensors.items())
    tensor_names, tensors = tuple(tensor_names), tuple(tensors)
    if len(named_scalars) > 0:
        scalar_names, scalars = zip(*named_scalars.items())
        scalar_names, scalars = tuple(scalar_names), tuple(scalars)
    else:
        scalar_names, scalars = tuple(), tuple()
    # get device and dtype
    t0 = tensors[0]
    device, dtype = t0.device, t0.dtype

    shapes = (t.shape for t in tensors)
    strides = (t.strides for t in tensors)
    # broadcast shape and compute strides
    shape = map(max, zip_longest(*map(reversed, shapes), fillvalue=1))
    shape = tuple(map(i32, shape))[::-1]
    ndim, numel = len(shape), reduce(lambda x, y: x * y, shape, 1)

    # create output tensors if necessary
    for out in output:
        if out not in tensor_names:
            tensor_names += (out,)
            tensors += (device.Tensor.empty(shape, dtype=dtype),)

    # build strides
    strides = tuple(
        (i32(0),) * (ndim - len(t.strides)) + tuple(map(lambda st_sh: i32(st_sh[0] if st_sh[1] > 1 else 0), zip(t.strides, t.shape)))
        for t in tensors
    )
    # collapse contiguous dimensions to minimize index computations in kernel
    if ndim > 1:
        shape, strides = _collapse_contiguous_dims(shape, strides)
        ndim = len(shape)

    # by default we read only tensors that are not in output
    read = tuple(n for n in tensor_names if n not in output) + additional_read
    buffer_dtypes = tuple(map(lambda t: t.dtype, tensors))
    scalar_dtypes = tuple(map(lambda s: np.dtype(type(s)), scalars))
    # build kernel and set arguments
    knl = build_atom_kernel(device.context,
        op=op,
        buffers=tensor_names,
        buffer_dtypes=buffer_dtypes,
        scalars=scalar_names,
        scalar_dtypes=scalar_dtypes,
        ndim=ndim,
        read=read,
        write=output
    )
    knl.set_args(
        *(t.data for t in tensors),          # buffers
        *(t.type(s) for t, s in zip(scalar_dtypes, scalars)),    # scalars
        *shape, *chain(*strides),            # shapes and strides
        *(i32(t.offset) for t in tensors),   # offsets
        i32(numel)                           # number of elements to compute
    )
    # execute kernel and return output tensors
    cl.enqueue_nd_range_kernel(device.queue, knl, [ceil(numel/block_size)*block_size], [block_size]).wait()    
    return tuple(t for n, t in zip(tensor_names, tensors) if n in output)

#
#   Matrix Multiplication Kernel
#

@lru_cache()
def cache_build_dot_kernel(context, 
    # data types
    dtype_A:str, 
    dtype_B:str, 
    dtype_O:str, 
    # kernel
    block_size:int,         # local block size
    work_per_thread:int     # number of elements each thread computes
) -> cl.Kernel:
    # convert dtypes to ctypes
    ctype_A = dtype_to_ctype(dtype_A)
    ctype_B = dtype_to_ctype(dtype_B)
    ctype_O = dtype_to_ctype(dtype_O)
    # build program
    return cl.Program(context, f"""
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
            // global batch index
            const uint b = get_global_id(0);
            // local thread index
            const uint li = get_local_id(1);
            const uint lj = get_local_id(2);
            // work-group index
            const uint bi = get_group_id(1);
            const uint bj = get_group_id(2);

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
            A_off += bi * TS * K + b * M * K;
            B_off += bj * TS + b * K * N;
            
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
            const uint O_off = b * M * N;
            const uint i = bi * TS + li;
            const uint j = bj * TS + lj;
            for (int wi = 0; wi < WPT; wi++)
                for (int wj = 0; wj < WPT; wj++)
                    O[(i + wj*RTS) * N + (j + wi*RTS) + O_off] = acc[wi][wj];
        }}
    """).build().matmul

def _match_blocks(T, block_size):
    B, M, N = T.shape
    if (M % block_size != 0) or (N % block_size != 0):
        shape = (B, ceil(M / block_size) * block_size, ceil(N / block_size) * block_size)
        T_pad = T.device.Tensor.zeros(shape, dtype=T.dtype)
        T_pad[:, :M, :N] = T
        return T_pad
    return T

def dot(
    # inputs
    X:OpenCLTensor, 
    Y:OpenCLTensor, 
    # kernel information
    block_size:int =8*16, 
    work_per_thread:int =8
) -> OpenCLTensor:
    assert 3 >= len(X.shape) == len(Y.shape) >= 2
    assert X.shape[:-2] == Y.shape[:-2]
    assert X.shape[-1] == Y.shape[-2]
    # get tensor information
    device = X.device
    n, M, N, K = len(X.shape), X.shape[-2], Y.shape[-1], X.shape[-1]
    # flatten batch dimensions
    X = X.reshape(-1, M, K)
    Y = Y.reshape(-1, K, N)
    assert X.shape[0] == Y.shape[0], "Batches do not align! (%i != %i)" % (X.shape[0], Y.shape[0])
    # pad inputs to be multiple of block size in both directions
    X = _match_blocks(X, block_size)
    Y = _match_blocks(Y, block_size)
    # create output tensor
    B, pad_M, pad_N, pad_K = X.shape[0], X.shape[1], Y.shape[2], X.shape[2]
    B, pad_M, pad_N, pad_K = i32(B), i32(pad_M), i32(pad_N), i32(pad_K)
    O = device.Tensor.empty(shape=(B, pad_M, pad_N), dtype=X.dtype) # TODO: broadcast dtype
    # kernel global and local thread layout
    global_shape = [B, pad_M // work_per_thread, pad_N // work_per_thread]
    local_shape = [1] + [block_size // work_per_thread] * 2
    # build and call kernel
    knl = cache_build_dot_kernel(device.context, X.dtype, Y.dtype, O.dtype, block_size, work_per_thread)
    e = knl(device.queue, global_shape, local_shape, 
            X.contiguous().data, Y.contiguous().data, O.data, 
            i32(X.offset), i32(Y.offset),
            pad_M, pad_N, pad_K)
    e.wait()
    # remove padding from output
    idx = (slice(0, B) if (n == 3) else 0, slice(0, M), slice(0, N))
    return O[idx]

@lru_cache()
def cache_build_reduction_kernel(context, 
    reduction:str,          # reduction operation
    # data information
    dtype:str,              # data type
    neutral:str,            # neural element of reduction
    ndim:int,               # dimension of input tensor
    use_strides:bool,       # use strided indices
    # kernel information
    block_size:int          # local block size
) -> cl.Kernel:
    ctype = dtype_to_ctype(dtype)
    # there seems to be some weird difference in the ussage of barriers between CPUs and GPUs
    is_gpu = (context.devices[0].get_info(cl.device_info.TYPE) == cl.device_type.GPU)
    # additional arguments list
    nl = lambda i: '\n' + ' ' * i
    # helper source
    get_idx = lambda idx, i: f"""uint {idx} = 0;
            {{
                uint pos, j = {i};
                {nl(16).join([
                    ("pos = j %% size_%i;" % d) + nl(16) +
                    ("%s += pos * stride_%i;" % (idx, d)) + 
                    ((nl(16) + "j /= size_%i;" % d) if d != 0 else "")
                    for d in range(ndim-1, -1, -1)
                ])}
            }}
    """ if use_strides else f"uint {idx} = {i};"
    # kernel source
    source = f"""
        void warpReduce(volatile __local {ctype}* sdata, uint tid) {{
            {ctype} a, b;
            a = sdata[tid]; b = sdata[tid + 32]; sdata[tid] = {reduction};
            a = sdata[tid]; b = sdata[tid + 16]; sdata[tid] = {reduction};
            a = sdata[tid]; b = sdata[tid + 8]; sdata[tid] = {reduction};
            a = sdata[tid]; b = sdata[tid + 4]; sdata[tid] = {reduction};
            a = sdata[tid]; b = sdata[tid + 2]; sdata[tid] = {reduction};
            a = sdata[tid]; b = sdata[tid + 1]; sdata[tid] = {reduction};
        }}

         __kernel void reduce ( 
             // input and output buffers
            __global const {ctype} *g_idata, 
            __global {ctype} *g_odata,
            // input buffer offset
            uint offset,
            // input shape and strides (only provided when needed)
            {' '.join(['const uint size_%i,' % i for i in range(ndim)]) if use_strides else ""}
            {' '.join(['const uint stride_%i,' % i for i in range(ndim)]) if use_strides else ""}
            // number of elements to reduce
            uint N
        ) {{
            uint tid = get_local_id(1);
            uint ls = get_local_size(1);
            // compute indices
            uint i = get_group_id(1) * (ls * 2) + tid;
            uint gi =  get_global_id(0) * N + i;
            uint group_i = get_group_id(0) * get_num_groups(1) + get_group_id(1);

            // allocate local memory buffer
            __local {ctype} sdata[{block_size}];
            // get indices
            {get_idx("idxA", "gi")}
            {get_idx("idxB", "gi + ls")}
            // perform first level of reduction,
            // reading from global memory, writing to shared memory
            {ctype} a = (i < N)? g_idata[offset + idxA] : {neutral};
            {ctype} b = (i + ls < N)? g_idata[offset + idxB] : {neutral};
            sdata[tid] = {reduction};
            barrier(CLK_LOCAL_MEM_FENCE);

            // unrolled inner reduction loop
            // for GPUs we unroll warps separately
            {nl(12).join([
                f"if (tid < {2**i}) {{ {ctype} a = sdata[tid], b = sdata[tid + {2**i}]; sdata[tid] = {reduction};}} barrier(CLK_LOCAL_MEM_FENCE);" 
                for i in reversed(range(6 if is_gpu else 0, int(log2(block_size))))
            ])}

            {(
                "// unroll warp reduction" + nl(12) +
                "if (tid < 32) warpReduce(sdata, tid);"
            ) if is_gpu else ""
            }

            // save partial reduction result
            if (tid == 0) g_odata[group_i] = sdata[0];
        }}
    """
    # print(source)
    return cl.Program(context, source).build().reduce

def reduction(
    reduction:str,  # reduction expression using variables 'a' and 'b'
    # input tensor
    T:OpenCLTensor,
    # options
    axis:Union[Tuple[int], int, None] =None,
    keepdims:bool =False,
    neutral:str ="0",
    # kernel information
    group_size:int =128
) -> OpenCLTensor:
    # get device
    device = T.device
    # prepare reduction axes
    axes = tuple(range(len(T.shape))) if axis is None else (axis,) if not isinstance(axis, tuple) else axis
    axes = tuple(i if i >= 0 else (len(T.shape) + i) for i in axes)
    # total number of elements to reduce
    reduce_numel = reduce(lambda x, y: x * y, (T.shape[i] for i in axes))
    keep_numel = reduce(lambda x, y: x * y, (T.shape[i] for i in range(len(T.shape)) if i not in axes), 1)
    n_work_groups = ceil(reduce_numel / (group_size * 2))  # number of work-groups needed

    # build output tensor
    shape = tuple(s if i not in axes else 1 for i, s in enumerate(T.shape) if (i not in axes) or keepdims)
    shape = (1,) if len(shape) == 0 else shape
    # output tensor also stores partial sums of each iterations, thus n_work_groups
    O = device.Tensor.empty(shape + (n_work_groups,), dtype=T.dtype)

    # transpose to have reduction dimensions at last
    if len(axes) < len(T.shape):
        perm = list(range(len(T.shape)))
        for i, j in enumerate(axes, 1):
            perm[-i], perm[j] = perm[j], perm[-i]
        T = T.transpose(*perm)

    # build kernels
    use_strides = (len(axes) < len(T.shape) and not T.is_contiguous())
    knl = cache_build_reduction_kernel(device.context, 
        reduction=reduction, 
        dtype=T.dtype, 
        neutral=neutral, 
        ndim=len(T.shape) if use_strides else 0,    # set to 0 if not needed to prevent compiling a new kernel
        use_strides=use_strides, 
        block_size=group_size
    )
    next_knl = cache_build_reduction_kernel(device.context, 
        reduction=reduction, 
        dtype=T.dtype, 
        neutral=neutral, 
        ndim=0, 
        use_strides=False, 
        block_size=group_size
    )

    # build additional strided input arguments
    stride_args = (
        *(i32(s) for s in T.shape),
        *(i32(st) for st in T.strides)
    ) if use_strides else tuple()

    while (reduce_numel > 1):
        knl.set_args(T.data, O.data, i32(T.offset), *stride_args, i32(reduce_numel))
        e = cl.enqueue_nd_range_kernel(device.queue, knl, [keep_numel, n_work_groups * group_size], [1, group_size])
        # update values
        T = O   # input of further iterations is output of current iteration
        reduce_numel = n_work_groups
        n_work_groups = ceil(reduce_numel / (group_size * 2))
        knl = next_knl
        stride_args = tuple()

    # wait for queue to finish
    e.wait()
    # remove partial sums stored in last dimension of output
    return device.Tensor(O.data, shape=shape, dtype=O.dtype)

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

@lru_cache()
def _cache_build_reduction_kernel(ctx, operation_str:str, ctype:str, neutral:str, use_strides:bool, block_size:int):
    # additional arguments list
    stride_args = ["__global uint* strides", "__global uint* shape", "uint dim"] if use_strides else []
    nl = '\n'
    # build kernel
    return cl.Program(ctx, f"""
        __kernel void reduce(
            uint n_red_items,
            __global {ctype}* T,
            __global {ctype}* O,
            const uint T_off{(',' + nl + ' ' * 12).join([""] + stride_args)}
        ) {{
            uint gi = get_global_id(0);
            uint gj = get_global_id(1);
            uint lj = get_local_id(1);

            __local {ctype} loc_buf[{block_size}];

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

def _reduction(operation_str:str, T, axis:int, keepdims:bool, neutral:str ="0", group_size:int =128):
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
    first_knl = cache_build_reduction_kernel(device.context, operation_str, ctype, neutral, use_strides=use_strides, block_size=group_size)
    further_knl = cache_build_reduction_kernel(device.context, operation_str, ctype, neutral, use_strides=False, block_size=group_size)
    # local memory
    loc_buf = cl.LocalMemory(T.dtype.itemsize * group_size)
    # stride arguments for first iteration
    stride_args = (
        cl.Buffer(device.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(T.strides, dtype=np.int32)),
        cl.Buffer(device.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.asarray(T.shape, dtype=np.int32)),
        np.int32(len(T.shape))
    ) if use_strides else tuple()

    # set kernel arguments
    first_knl.set_args(np.int32(n_red_items), T.data, O.data, np.int32(T.offset), *stride_args)
    # first iteration of reduction
    n, m = O.numel() // n_groups, ceil(n_red_items / group_size) * group_size
    e = cl.enqueue_nd_range_kernel(device.queue, first_knl, [n, m], [1, group_size])
    # further reduction iterations
    for i in range(1, n_iters):
        k = ceil(n_red_items / (group_size ** i))
        m = ceil(k / group_size) * group_size
        further_knl.set_args(np.int32(k), O.data, O.data, np.int32(0))
        e = cl.enqueue_nd_range_kernel(device.queue, further_knl, [n, m], [1, group_size])
    # wait for queue to finish
    e.wait()
    # remove partial sums stored in last dimension of output
    n = int(np.prod(shape[:-1])) * O.dtype.itemsize
    return device.Tensor(O.data, shape=shape[:-1], dtype=O.dtype)
