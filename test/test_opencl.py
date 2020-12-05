import unittest
# import lightgrad
from lightgrad.grad import CpuTensor
from lightgrad.grad import OpenCLTensor
from lightgrad.grad.opencl import Device
# set random seed
import numpy as np
np.random.seed(1337)

# get any device to use
device = Device.any()
if device.is_available():
    class Test_OpenCLTensor(unittest.TestCase):

        def compare_unary_func(self, cpu_f, opencl_f, shape=(3, 3), l=-1, h=1, eps=1e-3, transpose=False):
            # create random numpy array
            a = np.random.uniform(l, h, size=shape)
            a = a if not transpose else a.T
            # create cpu and opencl tensor and compare
            cpu_tensor = CpuTensor.from_numpy(a)
            opencl_tensor = device.Tensor.from_numpy(a)
            # apply functions
            cpu_out = cpu_f(cpu_tensor).numpy()
            opencl_out = opencl_f(opencl_tensor).numpy()
            # compare outputs
            np.testing.assert_allclose(opencl_out, cpu_out, rtol=1e-5, atol=1e-5)

        def compare_binary_func(self, cpu_f, opencl_f, a_shape=(3, 3), b_shape=(3, 3), l=-1, h=1, eps=1e-3, transpose=False):
            # create random numpy arrays
            a = np.random.uniform(l, h, size=a_shape) if not transpose else np.random.uniform(l, h, size=a_shape).T
            b = np.random.uniform(l, h, size=b_shape) if not transpose else np.random.uniform(l, h, size=b_shape).T
            # create cpu and opencl tensor and compare
            cpu_a, cpu_b = CpuTensor.from_numpy(a), CpuTensor.from_numpy(b)
            opencl_a, opencl_b = device.Tensor.from_numpy(a), device.Tensor.from_numpy(b)
            # apply functions
            cpu_out = cpu_f(cpu_a, cpu_b).numpy()
            opencl_out = opencl_f(opencl_a, opencl_b).numpy()
            # compare outputs
            np.testing.assert_allclose(opencl_out, cpu_out, rtol=1e-5, atol=1e-5)

        """ basic """

        def test_atom_kernel(self):
            from lightgrad.grad.opencl.ops import atom_kernel
            identity_kernel = lambda t: atom_kernel(t=t, out='o', operation_str='o=t')
            self.compare_unary_func(cpu_f=lambda t: t, opencl_f=identity_kernel)

        def test_atom_kernel_broadcast(self):
            from lightgrad.grad.opencl.ops import atom_kernel
            add_kernel = lambda a,b: atom_kernel(a=a, b=b, out='o', operation_str='o=a+b')
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(3, 3), b_shape=(1, 3))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(3, 3), b_shape=(3, 1))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(1, 3), b_shape=(3, 3))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(3, 1), b_shape=(3, 3))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(3, 1), b_shape=(1, 3))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(1, 3), b_shape=(3, 1))

        def test_atom_kernel_strides(self):
            from lightgrad.grad.opencl.ops import atom_kernel
            identity_kernel = lambda t: atom_kernel(t=t, out='o', operation_str='o=t')
            self.compare_unary_func(cpu_f=lambda t: t, opencl_f=identity_kernel, transpose=True)

        """ transformations """

        def test_transpose(self):
            f = lambda t: t.transpose(1, 0)
            self.compare_unary_func(f, f, transpose=False)
            self.compare_unary_func(f, f, transpose=True)
        def test_reshape(self):
            f = lambda t: t.reshape(t.numel())
            self.compare_unary_func(f, f, transpose=False)
            self.compare_unary_func(f, f, transpose=True)

        """ unary operators """
        def test_neg(self):
            self.compare_unary_func(CpuTensor.neg, OpenCLTensor.neg)

        """ Reductions/Selections """
            
        """ binary operators """
        def test_add(self):
            self.compare_binary_func(CpuTensor.add, OpenCLTensor.add)
        def test_sub(self):
            self.compare_binary_func(CpuTensor.sub, OpenCLTensor.sub)
        def test_mul(self):
            self.compare_binary_func(CpuTensor.mul, OpenCLTensor.mul)
        def test_div(self):
            self.compare_binary_func(CpuTensor.div, OpenCLTensor.div, l=0.1, h=10)
            self.compare_binary_func(CpuTensor.div, OpenCLTensor.div, l=-10, h=-0.1)
        def test_pow(self):
            self.compare_binary_func(CpuTensor.pow, OpenCLTensor.pow, l=0, h=1)
            
        """ more complex operations """

else:
    # device not available
    print("Not Testing OpenCL Tensor because no opencl devices were found!")


if __name__ == '__main__':
    unittest.main(verbose=2)