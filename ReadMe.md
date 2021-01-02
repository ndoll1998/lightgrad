# LightGrad
![Unit Tests](https://github.com/ndoll1998/lightgrad/workflows/Unit%20Tests/badge.svg)

A lightweight automatic differentiation library with some simple machine learning concepts build on top of it.

## How to use

We provide an API very simmilar to [pytorch](https://github.com/pytorch/pytorch). See the following simple example on how to use the autograd system (from `examples/gradient_descent.py`):

```python
import lightgrad as light

# create tensors
a = light.uniform(-1, 1, shape=(10, 10))
b = light.uniform(-1, 1, shape=(10, 10))
c = light.uniform(-1, 1, shape=(10, 10))
# objective to minimize
f = lambda: (a.tanh() + b.sigmoid()) @ (c.relu() - a.sigmoid())
# optimization loop
for epoch in range(100):
    # execute and compute gradients
    # the argument allow_fill allows for backpropagation 
    # starting from non-item tensors
    y = f()
    y.backward(allow_fill=True)

    # we need to disable gradients for the 
    # paremeter update
    with light.no_grad():
        # now we can easily access the gradients
        # and update the parameters
        a -= 0.1 * a.grad
        b -= 0.1 * b.grad
        c -= 0.1 * c.grad

    # reset gradients for next iteration
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()
```

### Optimizers
Alternatively to manually updating the parameters, we provides an optimizer module called `optim` with some standard optimizers already implemented (`SGD`, `Adam`, etc.). Using this we can reduce the above example to the following:

```python
...
# create optimizer
optim = light.optim.SGD([a, b, c], lr=0.1)
# optimization loop
for epoch in range(100):
    # use the optimizer for gradient resetting 
    # and parameter updates
    optim.zero_grad()
    f().backward(allow_fill=True)
    optim.step()
```

### Modules
Modules yield an easy way of managing parameters and combining them with their functionality. Additionally they provides furhter features for easy parameter manipulation (loading, mapping, etc.). We can simplify our example using the `Module` class as follows:

```python
import lightgrad as light
# define a module class holding all the neccessary parameters
class SimpleModule(light.nn.Module):
    """ Simple Module holding a few parameters """
    def __init__(self):
        nn.Module.__init__(self)
        self.a = light.uniform(-1, 1, shape=(10, 10))
        self.b = light.uniform(-1, 1, shape=(10, 10))
        self.c = light.uniform(-1, 1, shape=(10, 10))
    def forward(self):
        return (self.a.tanh() + self.b.sigmoid()) @ (self.c.relu() - self.a.sigmoid())
# create module and optimizer
f = SimpleModule()
optim = light.optim.SGD(f.parameters(), lr=0.1)
...
```

### Neural Networks
Combining the autograd system with both Optimizers and Modules results in a simple interface for defining, training and evaluating neural networks. The `lightgrad.nn` package already provides some common neural network components (`Linear`, `Conv2d`, etc.). A simple example of a convolutional neural network is shown below (from `examples/mnist.py`).
```python
import lightgrad.nn as nn

class CNN(nn.Module):
    """ Simple convolutional neural network """
    def __init__(self):
        nn.Module.__init__(self)
        # create convolutional layers
        self.c1 = nn.Conv2d(1, 8, kernelsize=3)
        self.c2 = nn.Conv2d(8, 16, kernelsize=3)
        # create classification layer
        self.l1 = nn.Linear(5 * 5 * 16, 10)
    def forward(self, x):
        # apply convolutions
        y = self.c1(x).max_pool().relu()
        y = self.c2(y).max_pool().relu()
        # classify
        y = self.l1(y.reshape(-1, 5 * 5 * 16))
        return y
```

---

## Accelerators
Lightgrad currently supports the following backends:
 - CPU (Default)
 - OpenCL

Pushing a tensor to another accelerator can be done using one function call.
```python
# create a cpu tensor and push it to opencl
light.uniform(-1, 1, shape=(10, 10)).opencl()
```
This can be advanced to pushing all parameters of a module to an accelerator by using the mapping functionality of modules.
```python
# push each parameter of the module to opencl
module.map_parameters(lambda p: p.opencl())
```

### Implement an Accelerator

Create a new Tensor class inheriting from `autograd.tensor.AbstractTensor`. The `AbstractTensor` takes care of all gradient managment. 

```python
import numpy as np
from lightgrad.autograd.tensor import AbstractTensor

class NewTensor(AbstractTensor):
    """ Tensor class for new accelerator """

    def __init__(self, *args, dtype:type =np.float32, requires_grad:bool =True):
        # create raw tensor data and initialize abstract tensor
        AbstractTensor.__init__(self, data=data, requires_grad=requires_grad)

    @property
    def dtype(self):
        raise NotImplementedError()
    @property
    def shape(self) -> tuple:
        raise NotImplementedError()

    @staticmethod
    def empty(shape, *args, **kwargs) -> "NewTensor":
        raise NotImplementedError()
    @staticmethod
    def zeros(shape, *args, **kwargs) -> "NewTensor":
        raise NotImplementedError()
    @staticmethod
    def ones(shape, *args, **kwargs) -> "NewTensor":
        raise NotImplementedError()
    @staticmethod
    def uniform(low, high, shape, *args, **kwargs) -> "NewTensor":
        raise NotImplementedError()
    @staticmethod
    def from_numpy(a:np.ndarray, requires_grad:bool =True) -> "NewTensor":
        raise NotImplementedError()
    def numpy(self) -> np.ndarray:
        raise NotImplementedError()
    def copy(self, requires_grad:bool =True) -> "NewTensor":
        raise NotImplementedError()
```

Further functionality is provided by registering subclasses of `lightgrad.autograd.func.Function` to the new tensor class.

```python
from lightgrad.autograd.func import Function

@NewTensor.register_op(name="exp")
class exp(Function):
    def forward(ctx, t):
        # compute exp on raw tensor data
        raw_x_data = t.data
        raw_y_data = raw_x_data.exp()
        # save result for backpropagation
        ctx.save_for_backward(raw_y_data)
        # return tensor instance holding raw data
        return NewTensor(raw_y_data)
    def backward(ctx, out_grad):
        # get raw data
        raw_out_grad_data = out_grad.data
        raw_y_data, = ctx.get_saved_tensors()
        # compute gradient of input
        raw_in_grad_data = raw_y_data * raw_out_grad_data
        return NewTensor(raw_in_grad_data)
```
The minimum set of operations needed for the autograd system are
```python
neg, add, mul, pow, fill
```
