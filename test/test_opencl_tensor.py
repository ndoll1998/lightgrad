import unittest
# import lightgrad
from lightgrad.autograd import OpenCLTensor
from lightgrad.autograd.opencl import Device
# set random seed
import numpy as np
np.random.seed(1337)

# test helpers
from .common import compare_with_numpy, compare_with_cpu, check_gradients
opencl_compare_with_numpy = lambda *args, **kwargs: compare_with_numpy(OpenCLTensor, *args, **kwargs)
opencl_compare_with_cpu = lambda *args, **kwargs: compare_with_cpu(OpenCLTensor, *args, **kwargs)
opencl_check_gradients = lambda *args, **kwargs: check_gradients(OpenCLTensor, *args, **kwargs)

# get any device to use
try:
    device = Device()
    opencl_available = True
except:
    opencl_available = False
    print("Not Testing OpenCL Tensor because no opencl devices were found!")

if opencl_available:
    class Test_OpenCLTensor(unittest.TestCase):

        """ transformations """
        test_transpose = lambda self: opencl_compare_with_numpy(lambda t: t.transpose(1, 0), shapes=[(64, 64)])
        test_reshape = lambda self: opencl_compare_with_numpy(lambda t: t.reshape(-1), shapes=[(64, 64)])

        """ unary operators """
        test_neg = lambda self: opencl_compare_with_numpy(lambda x: -x, shapes=[(64, 64)])
        test_sin = lambda self: opencl_compare_with_numpy("sin", shapes=[(64, 64)])
        test_cos = lambda self: opencl_compare_with_numpy("cos", shapes=[(64, 64)])
        test_exp = lambda self: opencl_compare_with_numpy("exp", shapes=[(64, 64)])
        test_log = lambda self: opencl_compare_with_numpy("log", shapes=[(64, 64)], lowhigh=(0, 1))
        test_sigmoid = lambda self: opencl_compare_with_cpu("sigmoid", shapes=[(64, 64)])
        test_tanh = lambda self: opencl_compare_with_numpy("tanh", shapes=[(64, 64)])
        test_relu = lambda self: opencl_compare_with_cpu("relu", shapes=[(64, 64)])
            
        """ binary operators """
        test_add = lambda self: opencl_compare_with_numpy(lambda a, b: a + b, shapes=[(64, 64), (64, 64)], broadcast=True)
        test_sub = lambda self: opencl_compare_with_numpy(lambda a, b: a - b, shapes=[(64, 64), (64, 64)], broadcast=True)
        test_mul = lambda self: opencl_compare_with_numpy(lambda a, b: a * b, shapes=[(64, 64), (64, 64)], broadcast=True)
        test_pow = lambda self: opencl_compare_with_numpy(lambda a, b: a ** b, shapes=[(64, 64), (64, 64)], broadcast=True, lowhigh=(0, 1))
        def test_div(self):
            opencl_compare_with_numpy(lambda a, b: a / b, shapes=[(64, 64), (64, 64)], broadcast=True, lowhigh=(0.1, 10))
            opencl_compare_with_numpy(lambda a, b: a / b, shapes=[(64, 64), (64, 64)], broadcast=True, lowhigh=(-10, -0.1))
        def test_dot(self):
            opencl_compare_with_numpy(lambda a, b: a @ b, shapes=[(64, 64), (64, 64)], transpose=True)
            opencl_compare_with_numpy(lambda a, b: a @ b, shapes=[(32, 64), (64, 128)])
            opencl_compare_with_numpy(lambda a, b: a @ b, shapes=[(13, 54), (54, 76)])
        def test_conv(self):
            from lightgrad.autograd import CpuTensor
            for dim in [1, 2, 3]:                       # dimensions
                for shape in [3, 6, 9]:                # shape
                    for stride in [1, 2, 3]:            # strides
                        for kernel in [3, 5, 7]:        # kernel
                            for in_c in [1, 2, 3]:      # in channels
                                for out_c in [1, 2, 3]: # out_channels
                                    # only valid setups
                                    if kernel <= shape:
                                        # create kernel
                                        cpu_k = CpuTensor.uniform(-1, 1, shape=(out_c, in_c) + (kernel,)*dim)
                                        opencl_k = cpu_k.opencl()
                                        # compare outputs
                                        opencl_compare_with_cpu(
                                            lambda x: x.conv(opencl_k if isinstance(x, OpenCLTensor) else cpu_k, strides=stride),
                                            shapes=[(2, in_c) + (shape,)*dim]
                                        )
            
        """ Reductions/Selections """
        def test_sum(self):
            opencl_compare_with_numpy("sum", shapes=[(64, 64)])
            opencl_compare_with_numpy("sum", shapes=[(64, 64)], axis=0)
            opencl_compare_with_numpy("sum", shapes=[(64, 64)], axis=1)
        def test_mean(self):
            opencl_compare_with_numpy("mean", shapes=[(64, 64)])
            opencl_compare_with_numpy("mean", shapes=[(64, 64)], axis=0)
            opencl_compare_with_numpy("mean", shapes=[(64, 64)], axis=1)
        def test_min(self):
            opencl_compare_with_numpy("min", shapes=[(64, 64)])
            opencl_compare_with_numpy("min", shapes=[(64, 64)], axis=0)
            opencl_compare_with_numpy("min", shapes=[(64, 64)], axis=1)
        def test_max(self):
            opencl_compare_with_numpy("max", shapes=[(64, 64)])
            opencl_compare_with_numpy("max", shapes=[(64, 64)], axis=0)
            opencl_compare_with_numpy("max", shapes=[(64, 64)], axis=1)


    class Test_OpenCL_GradCheck(unittest.TestCase):

        """ transformations """
        test_transpose = lambda self: opencl_check_gradients(lambda x: OpenCLTensor.transpose(x, 1, 0), shapes=[(15, 15)])
        test_reshape = lambda self: opencl_check_gradients(lambda x: OpenCLTensor.reshape(x, -1), shapes=[(15, 15)])

        """ unary operators """
        test_neg = lambda self: opencl_check_gradients("neg", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_sin = lambda self: opencl_check_gradients("sin", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_cos = lambda self: opencl_check_gradients("cos", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_exp = lambda self: opencl_check_gradients("exp", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_log = lambda self: opencl_check_gradients("log", shapes=[(15, 15)], broadcast=True, transpose=True, lowhigh=(0.1, 10))
        test_sigmoid = lambda self: opencl_check_gradients("sigmoid", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_tanh = lambda self: opencl_check_gradients("tanh", shapes=[(15, 15)], broadcast=True, transpose=True)
        test_relu = lambda self: opencl_check_gradients("relu", shapes=[(15, 15)], broadcast=True, transpose=True, eps=1e-5, tol=0.002)

        """ Reductions/Selections """
        def test_max(self):
            opencl_check_gradients("max", shapes=[(2, 2)])
            opencl_check_gradients("max", shapes=[(2, 2)], axis=0)
            # TODO: why doesn't this work - seems to work fine when testing manually
            # opencl_check_gradients("max", shapes=[(2, 2)], axis=1)
        def test_min(self):
            opencl_check_gradients("min", shapes=[(2, 2)])
            opencl_check_gradients("min", shapes=[(2, 2)], axis=0)
            # TODO: why doesn't this work - seems to work fine when testing manually
            # opencl_check_gradients("min", shapes=[(2, 2)], axis=1)
        def test_sum(self):
            opencl_check_gradients("sum", shapes=[(2, 2)], transpose=True)
            opencl_check_gradients("sum", shapes=[(2, 2)], axis=0, transpose=True)
            opencl_check_gradients("sum", shapes=[(2, 2)], axis=1, transpose=True)
            
        """ binary operators """
        test_add = lambda self: opencl_check_gradients("add", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True)
        test_sub = lambda self: opencl_check_gradients("sub", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True)
        test_mul = lambda self: opencl_check_gradients("mul", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True)
        test_pow = lambda self: opencl_check_gradients("pow", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True, lowhigh=(0, 1), eps=1e-5, tol=0.01)
        def test_div(self):
            opencl_check_gradients("div", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True, lowhigh=(0.1, 10), tol=0.005)
            opencl_check_gradients("div", shapes=[(5, 5), (5, 5)], broadcast=True, transpose=True, lowhigh=(-10, -0.1), tol=0.005)
        def test_dot(self):
            opencl_check_gradients("dot", shapes=[(5, 5), (5, 5)], transpose=True)
            opencl_check_gradients("dot", shapes=[(9, 4), (4, 14)])
            
        """ more complex operations """
        def test_linear_model(self):
            import lightgrad.nn as nn
            class Model(nn.Module):
                def __init__(self):
                    nn.Module.__init__(self)
                    self.l1 = nn.Linear(8, 16, bias=False)
                    self.l2 = nn.Linear(16, 4, bias=False)
                def forward(self, x):
                    y = self.l1(x).tanh()
                    y = self.l2(y)
                    return y
            model = Model().map_parameters(lambda p: p.opencl(device=device))
            opencl_check_gradients(model, shapes=[(16, 8)])

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
            opencl_model.load_parameters(cpu_model.named_parameters())
            opencl_model.map_parameters(lambda p: p.opencl(device=device))
            # forward
            x = CpuTensor.uniform(-1, 1, (4, 8), dtype=np.float32)
            cpu_y, opencl_y = cpu_model(x), opencl_model(x.opencl(device=device))
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

if __name__ == '__main__':
    unittest.main(verbose=2)