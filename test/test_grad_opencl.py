import unittest
# import lightgrad
from lightgrad.autograd import OpenCLTensor
from lightgrad.autograd.opencl import Device
from lightgrad.autograd.utils.gradcheck import assert_gradcheck
# set random seed
import numpy as np
np.random.seed(1337)

# get any device to use
device = Device.any()
if device.is_available():
    class OpenCLGradCheck(unittest.TestCase):

        def unary_func(self, f, shape=(3,), l=-1, h=1, eps=1e-3, transpose=False):
            t = np.random.uniform(l, h, size=shape).astype(np.float32)
            t = device.Tensor.from_numpy(t if not transpose else t.T)
            return assert_gradcheck(f, t, eps=eps)

        def simple_binary_func(self, f, a_shape=(3, 3), b_shape=(3, 3), l=-1, h=1, eps=1e-3, transpose=False):
            a = np.random.uniform(l, h, size=a_shape) if not transpose else np.random.uniform(l, h, size=a_shape).T
            b = np.random.uniform(l, h, size=b_shape) if not transpose else np.random.uniform(l, h, size=b_shape).T
            a = device.Tensor.from_numpy(a.astype(np.float32))
            b = device.Tensor.from_numpy(b.astype(np.float32))
            assert_gradcheck(lambda a: f(a, b), a, eps=eps)
            assert_gradcheck(lambda b: f(a, b), b, eps=eps)

        """ transformations """
        def test_transpose(self):
            self.unary_func(lambda t: t.transpose(0, 1), shape=(3, 2))
        def test_reshape(self):
            self.unary_func(lambda x: OpenCLTensor.reshape(x, -1))

        """ unary operators """
        def test_neg(self):
            self.unary_func(OpenCLTensor.neg)
        def test_sin(self):
            self.unary_func(OpenCLTensor.sin)
        def test_cos(self):
            self.unary_func(OpenCLTensor.cos)
        def test_exp(self):
            self.unary_func(OpenCLTensor.exp)
        def test_log(self):
            self.unary_func(OpenCLTensor.log, l=0.1, h=10)
        def test_sigmoid(self):
            self.unary_func(OpenCLTensor.sigmoid)
        def test_tanh(self):
            self.unary_func(OpenCLTensor.tanh)
        def test_relu(self):
            self.unary_func(OpenCLTensor.relu)

        """ Reductions/Selections """
            
        """ binary operators """
        def test_add(self):
            self.simple_binary_func(OpenCLTensor.add, transpose=False)
            self.simple_binary_func(OpenCLTensor.add, transpose=True)
        def test_sub(self):
            self.simple_binary_func(OpenCLTensor.sub, transpose=False)
            self.simple_binary_func(OpenCLTensor.sub, transpose=True)
        def test_mul(self):
            self.simple_binary_func(OpenCLTensor.mul, transpose=False)
            self.simple_binary_func(OpenCLTensor.mul, transpose=True)
        def test_div(self):
            self.simple_binary_func(OpenCLTensor.div, l=0.1, h=10, transpose=False)    # check positive values
            self.simple_binary_func(OpenCLTensor.div, l=-10, h=-0.1, transpose=False)  # also check for negatives
            self.simple_binary_func(OpenCLTensor.div, l=0.1, h=10, transpose=True)    # check positive values
            self.simple_binary_func(OpenCLTensor.div, l=-10, h=-0.1, transpose=True)  # also check for negatives
        def test_dot(self):
            self.simple_binary_func(OpenCLTensor.dot, a_shape=(3, 3), b_shape=(3, 3))
            self.simple_binary_func(OpenCLTensor.dot, a_shape=(7, 4), b_shape=(5, 7), transpose=True)
        def test_pow(self):
            self.simple_binary_func(OpenCLTensor.pow, l=0, h=1, transpose=False, eps=1e-4)
            self.simple_binary_func(OpenCLTensor.pow, l=0, h=1, transpose=True, eps=1e-4)
            
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
            model = Model().map_params(lambda p: p.opencl())
            self.unary_func(model, shape=(4, 8))

        def test_linear_model_compare_gradients(self):
            import lightgrad.nn as nn
            from lightgrad.autograd import CpuTensor
            class Model(nn.Module):
                def __init__(self):
                    nn.Module.__init__(self)
                    self.l1 = nn.Linear(8, 2, bias=False)
                    self.l2 = nn.Linear(2, 4, bias=False)
                def forward(self, x):
                    y = self.l1(x).tanh()
                    y = self.l2(y)
                    return y
            # create cpu and opencl model with same parameters
            cpu_model, opencl_model = Model(), Model()
            opencl_model.load_params(cpu_model.named_parameters())
            opencl_model.map_params(lambda p: p.opencl())
            # forward
            x = CpuTensor.uniform(-1, 1, (4, 8), dtype=np.float32)
            cpu_y, opencl_y = cpu_model(x), opencl_model(x.opencl())
            np.testing.assert_allclose(cpu_y.numpy(), opencl_y.numpy(), atol=5e-4, rtol=5e-4)
            # backpropagate
            cpu_y.backward(True)
            opencl_y.backward(True)
            # gather gradients
            cpu_grads = {n: p.grad.numpy() for n, p in cpu_model.named_parameters()}
            opencl_grads = {n: p.grad.numpy() for n, p in opencl_model.named_parameters()}
            # compare
            for n in cpu_grads.keys():
                cpu_g, opencl_g = cpu_grads[n], opencl_grads[n]
                np.testing.assert_allclose(cpu_g, opencl_g, atol=5e-4, rtol=5e-4)

else:
    # device not available
    print("Not Testing OpenCL Tensor because no opencl devices were found!")


if __name__ == '__main__':
    unittest.main(verbose=2)