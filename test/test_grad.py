import unittest
# import lightgrad
from lightgrad.grad import Tensor
from lightgrad.grad.gradcheck import gradcheck
# set random seed
import numpy as np
np.random.seed(1337)

class GradCheck(unittest.TestCase):

    def unary_func(self, f, shape=(3,), l=-1, h=1, eps=1e-3):
        t = Tensor.uniform(l, h, shape=shape)
        return self.assertTrue(gradcheck(f, t, eps=eps))

    def simple_binary_func(self, f, shape=(3, 3), l=-1, h=1, eps=1e-3):
        a = Tensor.uniform(l, h, shape=shape)
        b = Tensor.uniform(l, h, shape=shape)
        self.assertTrue(gradcheck(lambda a: f(a, b), a, eps=eps) and \
                        gradcheck(lambda b: f(a, b), b, eps=eps))

    """ unary operators """
    def test_neg(self):
        self.unary_func(Tensor.neg)
    def test_sin(self):
        self.unary_func(Tensor.sin)
    def test_cos(self):
        self.unary_func(Tensor.cos)
    def test_exp(self):
        self.unary_func(Tensor.exp)
    def test_log(self):
        self.unary_func(Tensor.log, l=0.1, h=10)
    def test_sigmoid(self):
        self.unary_func(Tensor.sigmoid)
    def test_tanh(self):
        self.unary_func(Tensor.tanh)
    # we use an gradient approximation for relu
    # def test_relu(self):
    #   self.unary_func(Tensor.relu, l=1, h=10)

    def test_max(self):
        self.unary_func(Tensor.max)
    def test_min(self):
        self.unary_func(Tensor.min)
        
    """ binary operators """
    def test_add(self):
        self.simple_binary_func(Tensor.add)
    def test_sub(self):
        self.simple_binary_func(Tensor.sub)
    def test_mul(self):
        self.simple_binary_func(Tensor.mul)
    def test_div(self):
        self.simple_binary_func(Tensor.div, l=0.1, h=10)    # check positive values
        self.simple_binary_func(Tensor.div, l=-10, h=-0.1)  # also check for negatives
    def test_dot(self):
        self.simple_binary_func(Tensor.mul, shape=(3, 3))        
    def test_pow(self):
        self.simple_binary_func(Tensor.pow, l=0, h=1)
        
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
        self.unary_func(Model(), shape=(4, 8))

if __name__ == '__main__':
    unittest.main(verbose=2)