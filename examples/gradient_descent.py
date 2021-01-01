import sys
sys.path.insert(0, "../")
import numpy as np
import lightgrad as light
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # create tensors
    a = light.uniform(-1, 1, shape=(10, 10), dtype=np.float32)
    b = light.uniform(-1, 1, shape=(10, 10), dtype=np.float32)
    c = light.uniform(-1, 1, shape=(10, 10), dtype=np.float32)
    # function to reduce
    f = lambda: (a.tanh() + b.sigmoid()) @ (c.relu() - a.sigmoid())

    # optimizer
    optim = light.optim.SGD([a, b, c], lr=0.1)

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
