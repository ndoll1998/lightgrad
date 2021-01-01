import numpy as np
from lightgrad.autograd.tensor import AbstractTensor
from lightgrad.autograd.utils.gradcheck import assert_gradcheck

def yield_input_pairs(cls,
    shapes:list,
    lowhigh:tuple =(-1, 1),
    dtype:type =np.float32,
    broadcast:bool =False,
    transpose:bool =False
) -> iter:
    # assertions
    assert len(lowhigh) == 2
    assert issubclass(cls, AbstractTensor)
    # yield default shapes
    np_arrays = [np.random.uniform(*lowhigh, size=shape).astype(dtype) for shape in shapes]
    cls_arrays = [cls.from_numpy(arr) for arr in np_arrays]
    yield np_arrays, cls_arrays
    # yield arrays for broadcasting
    if broadcast:
        # apply broadcasting over each dimension of each shape
        for i, shape in enumerate(shapes):
            for j in range(len(shape)):
                collapsed_shape = shape[:j] + (1,) + shape[j+1:]
                collapsed_np_array = np.random.uniform(*lowhigh, size=collapsed_shape)
                collapsed_cls_array = cls.from_numpy(collapsed_np_array)
                yield (
                    np_arrays[:i] + [collapsed_np_array] + np_arrays[i+1:],
                    cls_arrays[:i] + [collapsed_cls_array] + cls_arrays[i+1:]
                )
    # yield transposed arrays
    if transpose:
        for i, (np_array, cls_array, shape) in enumerate(zip(np_arrays, cls_arrays, shapes)):
            perm = list(reversed(range(len(shape))))
            yield (
                np_arrays[:i] + [np_array.transpose(*perm)] + np_arrays[i+1:],
                cls_arrays[:i] + [cls_array.transpose(*perm)] + cls_arrays[i+1:]
            )

def compare_with_numpy(cls, 
    fn_or_name:str, 
    shapes:list,
    lowhigh:tuple =(-1, 1),
    dtype:type =np.float32,
    broadcast:bool =False,
    transpose:bool =False,
    **kwargs
) -> None:
    """ Compare output with numpy """
    # get functions for numpy and tensor class
    if isinstance(fn_or_name, str):
        np_fn = getattr(np, fn_or_name)
        cls_fn = getattr(cls, fn_or_name)
    else:
        np_fn, cls_fn = fn_or_name, fn_or_name
    # test for all pairs of inputs
    array_pair_iter = yield_input_pairs(cls, 
        shapes=shapes, 
        lowhigh=lowhigh, 
        dtype=dtype, 
        broadcast=broadcast, 
        transpose=transpose
    )
    for np_arrays, cls_arrays in array_pair_iter:
        # apply function
        np_out = np_fn(*np_arrays, **kwargs)
        cls_out = cls_fn(*cls_arrays, **kwargs).numpy()
        # assert all close
        np.testing.assert_allclose(np_out, cls_out, rtol=1e-5, atol=1e-5)

def compare_with_cpu(cls, 
    fn_or_name:str, 
    shapes:list,
    lowhigh:tuple =(-1, 1),
    dtype:type =np.float32,
    broadcast:bool =False,
    transpose:bool =False,
    **kwargs
) -> None:
    from lightgrad import CpuTensor
    """ Compare output with output of cpu tensor """
    # get functions for numpy and tensor class
    if isinstance(fn_or_name, str):
        cpu_fn = getattr(CpuTensor, fn_or_name)
        cls_fn = getattr(cls, fn_or_name)
    else:
        cpu_fn, cls_fn = fn_or_name, fn_or_name
    # test for all pairs of inputs
    array_pair_iter = yield_input_pairs(cls, 
        shapes=shapes, 
        lowhigh=lowhigh, 
        dtype=dtype, 
        broadcast=broadcast, 
        transpose=transpose
    )
    for np_arrays, cls_arrays in array_pair_iter:
        cpu_arrays = [CpuTensor.from_numpy(arr) for arr in np_arrays]
        # apply function
        cpu_out = cpu_fn(*cpu_arrays, **kwargs).numpy()
        cls_out = cls_fn(*cls_arrays, **kwargs).numpy()
        # assert all close
        np.testing.assert_allclose(cpu_out, cls_out, rtol=1e-5, atol=1e-5)

def check_gradients(cls,
    fn_or_name:str,
    shapes:list,
    lowhigh:tuple =(-1, 1),
    dtype:type =np.float32,
    broadcast:bool =False,
    transpose:bool =False,
    eps:float =1e-3,
    tol=5e-4,
    **kwargs
) -> None:
    # get functions for numpy and tensor class
    fn = getattr(cls, fn_or_name) if isinstance(fn_or_name, str) else fn_or_name
    # test for all pairs of inputs
    array_pair_iter = yield_input_pairs(cls, 
        shapes=shapes, 
        lowhigh=lowhigh, 
        dtype=dtype, 
        broadcast=broadcast,
        transpose=transpose
    )
    for _, cls_arrays in array_pair_iter:
        # check gradient for each input tensor
        for i, arr in enumerate(cls_arrays):
            f = lambda x: fn(*cls_arrays[:i], x, *cls_arrays[i+1:], **kwargs)
            assert_gradcheck(f=f, x=arr, eps=eps, atol=tol, rtol=tol)