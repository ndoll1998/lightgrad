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
    def test_transpose(self):
        cpu_check_gradients(CpuTensor.transpose, shapes=[(45, 65)])
    def test_reshape(self):
        cpu_check_gradients(lambda x: CpuTensor.reshape(x, -1), shapes=[(45, 65)])
    def test_pad(self):
        cpu_check_gradients(lambda x: CpuTensor.pad(x, padding=2), shapes=[(45, 65)])

    """ unary operators """
    def test_neg(self):
        cpu_check_gradients(CpuTensor.neg, shapes=[(10, 15)])
    def test_sin(self):
        cpu_check_gradients(CpuTensor.sin, shapes=[(10, 15)])
    def test_cos(self):
        cpu_check_gradients(CpuTensor.cos, shapes=[(10, 15)])
    def test_exp(self):
        cpu_check_gradients(CpuTensor.exp, shapes=[(10, 15)])
    def test_log(self):
        cpu_check_gradients(CpuTensor.log, shapes=[(10, 15)], lowhigh=(0.1, 10))
    def test_sigmoid(self):
        cpu_check_gradients(CpuTensor.sigmoid, shapes=[(10, 15)])
    def test_tanh(self):
        cpu_check_gradients(CpuTensor.tanh, shapes=[(10, 15)])
    def test_relu(self):
        cpu_check_gradients(CpuTensor.relu, shapes=[(10, 15)], eps=1e-5, tol=0.002)

    """ Reductions/Selections """
    def test_max(self):
        cpu_check_gradients(CpuTensor.max, shapes=[(10, 15)])
    def test_min(self):
        cpu_check_gradients(CpuTensor.min, shapes=[(10, 15)])
        
    """ binary operators """
    def test_add(self):
        cpu_check_gradients(CpuTensor.add, shapes=[(10, 15), (10, 15)], broadcast=True)
    def test_sub(self):
        cpu_check_gradients(CpuTensor.sub, shapes=[(10, 15), (10, 15)], broadcast=True)
    def test_mul(self):
        cpu_check_gradients(CpuTensor.mul, shapes=[(10, 15), (10, 15)], broadcast=True)
    def test_div(self):
        cpu_check_gradients(CpuTensor.div, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(0.1, 10), tol=5e-3)
        cpu_check_gradients(CpuTensor.div, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(-10, -0.1), tol=5e-3)
    def test_pow(self):
        cpu_check_gradients(CpuTensor.pow, shapes=[(10, 15), (10, 15)], broadcast=True, lowhigh=(1, 2), eps=1e-5, tol=0.01)
    def test_dot(self):
        cpu_check_gradients(CpuTensor.dot, shapes=[(10, 15), (15, 10)])
    def test_convolution(self):
        cpu_check_gradients(CpuTensor.conv, shapes=[(3, 2, 5, 5), (4, 2, 3, 3)], strides=1)
        
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