import unittest
# import lightgrad
from lightgrad.grad import CpuTensor
from lightgrad.grad import OpenCLTensor
# set random seed
import numpy as np
np.random.seed(1337)

class Test_OpenCLTensor(unittest.TestCase):

    def compare_unary_func(self, cpu_f, opencl_f, shape=(3,), l=-1, h=1, eps=1e-3):
        # create cpu and opencl tensor and compare
        cpu_tensor = CpuTensor.uniform(l, h, shape=shape)
        opencl_tensor = cpu_tensor.opencl()
        # apply functions
        cpu_out = cpu_f(cpu_tensor).numpy()
        opencl_out = opencl_f(opencl_tensor).numpy()
        # compare outputs
        np.testing.assert_allclose(opencl_out, cpu_out)

    def compare_binary_func(self, cpu_f, opencl_f, a_shape=(3, 3), b_shape=(3, 3), l=-1, h=1, eps=1e-3):
        # create cpu and opencl tensor and compare
        cpu_a, cpu_b = CpuTensor.uniform(l, h, shape=a_shape), CpuTensor.uniform(l, h, shape=b_shape)
        opencl_a, opencl_b = cpu_a.opencl(), cpu_b.opencl()
        # apply functions
        cpu_out = cpu_f(cpu_a, cpu_b).numpy()
        opencl_out = opencl_f(opencl_a, opencl_b).numpy()
        # compare outputs
        np.testing.assert_allclose(opencl_out, cpu_out)

    """ transformations """

    """ unary operators """
    def test_neg(self):
        self.compare_unary_func(CpuTensor.neg, OpenCLTensor.neg)

    """ Reductions/Selections """
        
    """ binary operators """
    def test_add(self):
        self.compare_binary_func(CpuTensor.add, OpenCLTensor.add)
        # also test broadcasting
        self.compare_binary_func(CpuTensor.add, OpenCLTensor.add, a_shape=(1, 3))
        self.compare_binary_func(CpuTensor.add, OpenCLTensor.add, b_shape=(1, 3))
    def test_sub(self):
        self.compare_binary_func(CpuTensor.sub, OpenCLTensor.sub)
        # also test broadcasting
        self.compare_binary_func(CpuTensor.sub, OpenCLTensor.sub, a_shape=(1, 3))
        self.compare_binary_func(CpuTensor.sub, OpenCLTensor.sub, b_shape=(1, 3))
    def test_mul(self):
        self.compare_binary_func(CpuTensor.mul, OpenCLTensor.mul)
        # also test broadcasting
        self.compare_binary_func(CpuTensor.mul, OpenCLTensor.mul, a_shape=(1, 3))
        self.compare_binary_func(CpuTensor.mul, OpenCLTensor.mul, b_shape=(1, 3))
        
    """ more complex operations """

if __name__ == '__main__':
    unittest.main(verbose=2)