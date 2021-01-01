import sys
sys.path.insert(0, "../")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import lightgrad as light
import lightgrad.nn as nn
from lightgrad.autograd.utils.profiler import Profiler

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.c1 = nn.Conv2d(1, 8, kernelsize=3, bias=False, pad=0)
        self.c2 = nn.Conv2d(8, 16, kernelsize=3, bias=False, pad=0)
        self.l1 = nn.Linear(5 * 5 * 16, 10)
    def forward(self, x):
        y = self.c1(x).max_pool().relu()
        y = self.c2(y).max_pool().relu()
        y = self.l1(y.reshape(-1, 5 * 5 * 16))
        return y

class NN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(28 * 28, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)
    def forward(self, x):
        y = self.l1(x.reshape(-1, 28 * 28)).relu()
        y = self.l2(y)
        return y

if __name__ == '__main__':

    # device
    to_device = lambda t: t.opencl()
    # load datasets
    mnist_train = light.data.MNIST(train=True, shuffle=True, batchsize=128)
    mnist_test = light.data.MNIST(train=False, shuffle=False, batchsize=128)
    # create model
    model = NN().map_params(to_device)
    optim = light.optim.AdaBelief(model.parameters(), lr=0.001)

    steps = 200
    # train model
    with Profiler() as p:
        losses = []
        pbar = trange(steps)
        for i in pbar:
            # get a random data sample
            idx = np.random.randint(0, mnist_train.n, size=128)
            x, y_hat = mnist_train[idx]
            x = x.reshape(-1, 1, 28, 28).detach()
            # predict and compute error
            y = model(to_device(x))
            # l = cross_entropy(y, to_device(y_hat))
            one_hot = light.zeros((idx.shape[0], 10))
            one_hot[range(idx.shape[0]), y_hat] = 1
            l = light.loss.mse(y, to_device(one_hot))
            # optimize
            optim.zero_grad()
            l.backward()
            optim.step()
            # progress
            losses.append(l.item())
            pbar.set_postfix({'loss': sum(losses[-100:]) / min(100, len(losses))})

        print("\n")
        p.print()

    # evaluate model
    total_hits = 0
    for x, y_hat in mnist_test:
        y = model(to_device(x.reshape(x.shape[0], 1, 28, 28)))
        y = np.argmax(y.numpy(), axis=1)
        total_hits += (y == y_hat.data).sum()
    acc = total_hits / mnist_test.n
    print("Accuracy:\t", acc, end='\n\n')

    # infer model
    mnist_test.shuffle()
    x, target = mnist_test[0]
    predict = np.argmax(model(to_device(x.reshape(1, 1, 28, 28))).numpy(), axis=1)[0]
    print("Prediction:\t", predict, "\nTarget:\t\t", target.item())

    # visualize convolutions of CNN
    if hasattr(model, 'c1'):
        w = model.c1.w.numpy()
        f, axes = plt.subplots(w.shape[0], w.shape[1])
        axes = axes.reshape(w.shape[0], w.shape[1])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                axes[i, j].imshow(w[i, j, ...], cmap='gray')
        # plt.show()