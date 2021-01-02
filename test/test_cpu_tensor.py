import unittest
# import lightgrad and gradcheck
from lightgrad.autograd import CpuTensor
# set random seed
import numpy as np
np.random.seed(1234)

# test helpers
from .common import check_gradients
cpu_check_gradients = lambda *args, **kwargs: check_gradients(CpuTensor, *args, **kwargs)

class Test_Cpu_GradCheck(unittest.TestCase):

    """ transformations """
    test_transpose = lambda self: cpu_check_gradients(CpuTensor.transpose, shapes=[(45, 65)])
    test_reshape = lambda self: cpu_check_gradients(lambda x: CpuTensor.reshape(x, -1), shapes=[(45, 65)])
    test_pad = lambda self: cpu_check_gradients(lambda x: CpuTensor.pad(x, padding=2), shapes=[(45, 65)])

    """ unary operators """
    test_neg = lambda self: cpu_check_gradients(CpuTensor.neg, shapes=[(10, 15)])
    test_sin = lambda self: cpu_check_gradients(CpuTensor.sin, shapes=[(10, 15)])
    test_cos = lambda self: cpu_check_gradients(CpuTensor.cos, shapes=[(10, 15)])
    test_exp = lambda self: cpu_check_gradients(CpuTensor.exp, shapes=[(10, 15)])
    test_log = lambda self: cpu_check_gradients(CpuTensor.log, shapes=[(10, 15)], lowhigh=(0.1, 10))
    test_sigmoid = lambda self: cpu_check_gradients(CpuTensor.sigmoid, shapes=[(10, 15)])
    test_tanh = lambda self: cpu_check_gradients(CpuTensor.tanh, shapes=[(10, 15)])
    test_relu = lambda self: cpu_check_gradients(CpuTensor.relu, shapes=[(10, 15)], eps=1e-5, tol=0.002)

    """ Reductions/Selections """
    test_max = lambda self: cpu_check_gradients(CpuTensor.max, shapes=[(10, 15)])
    test_min = lambda self: cpu_check_gradients(CpuTensor.min, shapes=[(10, 15)])
        
    """ binary operators """
    test_add = lambda self: cpu_check_gradients(CpuTensor.add, shapes=[(10, 15), (10, 15)], broadcast=True)
    test_sub = lambda self: cpu_check_gradients(CpuTensor.sub, shapes=[(10, 15), (10, 15)], broadcast=True)
    test_mul = lambda self: cpu_check_gradients(CpuTensor.mul, shapes=[(10, 15), (10, 15)], broadcast=True)
    test_pow = lambda self: cpu_check_gradients(CpuTensor.pow, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(1, 2), eps=1e-5, tol=0.01)
    test_dot = lambda self: cpu_check_gradients(CpuTensor.dot, shapes=[(10, 15), (15, 10)])
    test_convolution = lambda self: cpu_check_gradients(CpuTensor.conv, shapes=[(3, 2, 5, 5), (4, 2, 3, 3)], strides=1)
    def test_div(self):
        cpu_check_gradients(CpuTensor.div, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(0.1, 10), tol=5e-3)
        cpu_check_gradients(CpuTensor.div, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(-10, -0.1), tol=5e-3)
        
    """ more complex operations """
    def test_linear_model(self):
        import lightgrad.nn as nn
        class Model(nn.Module):
            def __init__(self):
                nn.Module.__init__(self)
                self.l1 = nn.Linear(8, 16)
                self.l2 = nn.Linear(16, 4)
            def forward(self, x):
                y = self.l1(x).tanh()
                y = self.l2(y)
                return y
        cpu_check_gradients(Model(), shapes=[(16, 8)])

if __name__ == '__main__':
    unittest.main(verbose=2)