import sys
sys.path.insert(0, "../")
import numpy as np
import matplotlib.pyplot as plt
from lightgrad.grad import Tensor
from lightgrad.optim import SGD

if __name__ == '__main__':

    # create tensors
    a = Tensor(np.random.uniform(-1, 1, size=(10, 10)).astype(np.float32))
    b = Tensor(np.random.uniform(-1, 1, size=(10, 10)).astype(np.float32))
    c = Tensor(np.random.uniform(-1, 1, size=(10, 10)).astype(np.float32))
    # function to reduce
    f = lambda: b.sigmoid() @ c.relu()

    # optimizer
    optim = SGD([a, b], lr=0.1)

    def step():
        y = f()
        optim.zero_grad()
        y.backward(allow_fill=True)
        optim.step()
        return y.data.sum()
    ys = [step() for _ in range(100)]

    # plot
    plt.plot(ys)
    plt.xlabel("Steps")
    plt.ylabel("y")
    plt.show()
