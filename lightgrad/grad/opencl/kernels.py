import numpy as np
import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
import functools
from math import ceil

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

@functools.lru_cache()
def cache_build_dot_kernel(ctx, ctype_A:str, ctype_B:str, ctype_O:str, block_size:int):
    return cl.Program(ctx, f"""
        #define TS {block_size}

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
            __local {ctype_A} Asub[TS][TS];
            __local {ctype_B} Bsub[TS][TS];
            // accumulate
            {ctype_O} acc = 0;

            // read offsets
            const uint Aoff = bi * TS * K;
            const uint Boff = bj * TS;
            
            for (int t = 0; t < K; t+=TS) {{
                // load tile into local memory
                Asub[lj][li] = ((t+li<K) && (bi*TS+lj<M))? A[Aoff + lj * K + t + li] : 0;
                Bsub[lj][li] = ((t+lj<K) && (bj*TS+li)<N)? B[Boff + (t + lj) * N + li] : 0;
                barrier(CLK_LOCAL_MEM_FENCE);
                // accumulate
                for (int k=0; k < TS; k++)
                    acc += Asub[lj][k] * Bsub[k][li];
                // wait until work group finished loop
                barrier(CLK_LOCAL_MEM_FENCE);
            }}
            // write to output
            uint i = bi * TS + lj;
            uint j = bj * TS + li;
            if ((i < M) && (j < N)) {{ 
                O[i * N + j] = acc; 
            }}
        }}
    """).build()

def dot_kernel(A, B, block_size=16):
    assert len(A.shape) == len(B.shape) == 2
    assert A.shape[1] == B.shape[0]
    # tensor is tranposed iff it is not contiguous
    transpose_A = not A.is_contiguous
    transpose_B = not B.is_contiguous
    # create output tensor
    device = A.device
    M, N, K = np.int32(A.shape[0]), np.int32(B.shape[1]), np.int32(A.shape[1])
    O = device.Tensor.empty(shape=(M, N), dtype=A.dtype) # TODO: broadcast dtype
    # get data types
    ctype_A = dtype_to_ctype(A.dtype)
    ctype_B = dtype_to_ctype(B.dtype)
    ctype_O = dtype_to_ctype(O.dtype)
    # build and call kernel
    global_shape = [ceil(M/block_size)*block_size, ceil(N/block_size)*block_size]
    local_shape = [block_size] * 2
    prg = cache_build_dot_kernel(device.ctx, ctype_A, ctype_B, ctype_O, block_size)
    prg.matmul(device.queue, global_shape, local_shape, A.contiguous().data, B.contiguous().data, O.data, M, N, K)
    device.queue.finish()
    return O
    
