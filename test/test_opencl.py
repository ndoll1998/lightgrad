import unittest
# import lightgrad
from lightgrad.autograd import CpuTensor
from lightgrad.autograd import OpenCLTensor
from lightgrad.autograd.opencl import Device
# set random seed
import numpy as np
np.random.seed(1337)

# get any device to use
try:
    device = Device()
    opencl_available = True
except:
    opencl_available = False
    print("Not Testing OpenCL Tensor because no opencl devices were found!")

if opencl_available:
    class Test_OpenCLTensor(unittest.TestCase):

        def compare_unary_func(self, cpu_f, opencl_f, shape=(64, 64), l=-1, h=1, eps=1e-3, transpose=False):
            # create random numpy array
            a = np.random.uniform(l, h, size=shape).astype(np.float32)
            # create cpu and opencl tensor and compare
            cpu_tensor = CpuTensor.from_numpy(a)
            opencl_tensor = device.Tensor.from_numpy(a)
            if transpose:
                a = a.transpose(1, 0)
                cpu_tensor = cpu_tensor.transpose(1, 0)
                opencl_tensor = opencl_tensor.transpose(1, 0)
            # apply functions
            cpu_out = cpu_f(cpu_tensor).numpy()
            opencl_out = opencl_f(opencl_tensor).numpy()
            # compare outputs
            np.testing.assert_allclose(opencl_out, cpu_out, rtol=1e-5, atol=1e-5)

        def compare_binary_func(self, cpu_f, opencl_f, a_shape=(64, 64), b_shape=(64, 64), l=-1, h=1, eps=1e-3, transpose=False):
            # create random numpy arrays
            a = np.random.uniform(l, h, size=a_shape).astype(np.float32)
            b = np.random.uniform(l, h, size=b_shape).astype(np.float32)
            # create cpu and opencl tensor and compare
            cpu_a, cpu_b = CpuTensor.from_numpy(a), CpuTensor.from_numpy(b)
            opencl_a, opencl_b = device.Tensor.from_numpy(a), device.Tensor.from_numpy(b)
            if transpose:
                a, b = a.transpose(1, 0), b.transpose(1, 0)
                cpu_a, cpu_b = cpu_a.transpose(1, 0), cpu_b.transpose(1, 0)
                opencl_a, opencl_b = opencl_a.transpose(1, 0), opencl_b.transpose(1, 0)
            # apply functions
            cpu_out = cpu_f(cpu_a, cpu_b).numpy()
            opencl_out = opencl_f(opencl_a, opencl_b).numpy()
            # compare outputs
            np.testing.assert_allclose(opencl_out, cpu_out, rtol=1e-5, atol=1e-5)

        """ basic """

        def test_atom_kernel(self):
            from lightgrad.autograd.opencl.kernels import atom
            identity_kernel = lambda t: atom(t=t, output='o', op='o=t')[0]
            self.compare_unary_func(cpu_f=lambda t: t, opencl_f=identity_kernel)

        def test_atom_kernel_broadcast(self):
            from lightgrad.autograd.opencl.kernels import atom
            add_kernel = lambda a,b: atom(a=a, b=b, output='o', op='o=a+b')[0]
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(64, 64), b_shape=(1, 64))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(64, 64), b_shape=(64, 1))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(1, 64), b_shape=(64, 64))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(64, 1), b_shape=(64, 64))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(64, 1), b_shape=(1, 64))
            self.compare_binary_func(cpu_f=lambda a, b: a + b, opencl_f=add_kernel, a_shape=(1, 64), b_shape=(64, 1))

        def test_atom_kernel_strides(self):
            from lightgrad.autograd.opencl.kernels import atom
            identity_kernel = lambda t: atom(t=t, output='o', op='o=t')[0]
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
        def test_sin(self):
            self.compare_unary_func(CpuTensor.sin, OpenCLTensor.sin)
        def test_cos(self):
            self.compare_unary_func(CpuTensor.cos, OpenCLTensor.cos)
        def test_exp(self):
            self.compare_unary_func(CpuTensor.exp, OpenCLTensor.exp)
        def test_log(self):
            self.compare_unary_func(CpuTensor.log, OpenCLTensor.log, l=0.1, h=10)
        def test_sigmoid(self):
            self.compare_unary_func(CpuTensor.sigmoid, OpenCLTensor.sigmoid)
        def test_tanh(self):
            self.compare_unary_func(CpuTensor.tanh, OpenCLTensor.tanh)
        def test_relu(self):
            self.compare_unary_func(CpuTensor.relu, OpenCLTensor.relu)
            
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
        def test_dot(self):
            self.compare_binary_func(CpuTensor.dot, OpenCLTensor.dot, a_shape=(64, 64), b_shape=(64, 64))
            self.compare_binary_func(CpuTensor.dot, OpenCLTensor.dot, a_shape=(32, 64), b_shape=(128, 32), transpose=True)
            self.compare_binary_func(CpuTensor.dot, OpenCLTensor.dot, a_shape=(13, 54), b_shape=(54, 76))
        def test_pow(self):
            self.compare_binary_func(CpuTensor.pow, OpenCLTensor.pow, l=0, h=1)
            
        """ Reductions/Selections """
        def test_sum(self):
            self.compare_unary_func(CpuTensor.sum, OpenCLTensor.sum, shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.sum(t, axis=0), lambda t: OpenCLTensor.sum(t, axis=0), shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.sum(t, axis=1), lambda t: OpenCLTensor.sum(t, axis=1), shape=(64, 64))
        def test_mean(self):
            self.compare_unary_func(CpuTensor.mean, OpenCLTensor.mean, shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.mean(t, axis=0), lambda t: OpenCLTensor.mean(t, axis=0), shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.mean(t, axis=1), lambda t: OpenCLTensor.mean(t, axis=1), shape=(64, 64))
        def test_min(self):
            self.compare_unary_func(CpuTensor.min, OpenCLTensor.min, shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.min(t, axis=0), lambda t: OpenCLTensor.min(t, axis=0), shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.min(t, axis=1), lambda t: OpenCLTensor.min(t, axis=1), shape=(64, 64))
        def test_max(self):
            self.compare_unary_func(CpuTensor.max, OpenCLTensor.max, shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.max(t, axis=0), lambda t: OpenCLTensor.max(t, axis=0), shape=(64, 64))
            self.compare_unary_func(lambda t: CpuTensor.max(t, axis=1), lambda t: OpenCLTensor.max(t, axis=1), shape=(64, 64))

        """ more complex operations """


if __name__ == '__main__':
    unittest.main(verbose=2)