from lightgrad.autograd import Gradients, Tensor

class Optimizer(object):
    def __init__(self, parameters:iter) -> None:
        self.parameters = tuple(parameters)
        assert all((isinstance(p, Tensor) for p in self.parameters))
    def zero_grad(self) -> None:
        for p in self.parameters:
            p.zero_grad()
    @Gradients.no_grad()
    def step(self) -> None:
        for i, p in enumerate(self.parameters):
            p += self.compute_delta(p.grad, i)
    def compute_delta(self, grad:Tensor, idx:int) -> Tensor:
        raise NotImplementedError()

class SGD(Optimizer):
    """ Stochastic Gradient Descent """
    def __init__(self, parameters:iter, lr:float, momentum:float =0.0):
        Optimizer.__init__(self, parameters)
        self.prev_deltas = [0] * len(self.parameters)
        self.lr, self.momentum = lr, momentum
    def compute_delta(self, grad, i):
        self.prev_deltas[i] = -self.lr * grad + self.momentum * self.prev_deltas[i]
        return self.prev_deltas[i]

class Adam(Optimizer):
    """ ADAptive Moment estimation """
    def __init__(self, parameters:iter, lr:float, beta1:float =0.9, beta2:float =0.999, eps:float =1e-8):
        Optimizer.__init__(self, parameters)
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        # moment vectors
        self.m = [0] * len(self.parameters)
        self.v = [0] * len(self.parameters)
    def compute_delta(self, grad, i):
        self.t += 1
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * grad**2
        m, v = self.m[i] / (1 - self.b1**self.t), self.v[i] / (1 - self.b2**self.t)
        return -self.lr * m / (v**0.5 + self.eps)

class AdaBelief(Adam):
    """ Adapting Stepsizes by the Belief in Observed Gradients 
        Paper: https://arxiv.org/abs/2010.07468
    """
    def compute_delta(self, grad, i):
        self.t += 1
        self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * grad
        self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (grad - self.m[i])**2
        m, v = self.m[i] / (1 - self.b1**self.t), self.v[i] / (1 - self.b2**self.t)
        return -self.lr * m / (v**0.5 + self.eps)
