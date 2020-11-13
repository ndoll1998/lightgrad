import sys
sys.path.insert(0, "../")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from lightgrad import nn
from lightgrad.loss import cross_entropy
from lightgrad.optim import AdaBelief
from lightgrad.utils.data import MNIST_Train, MNIST_Test

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.c1 = nn.Conv2d(1, 2, kernelsize=5, bias=False)
        self.c2 = nn.Conv2d(2, 4, kernelsize=3, bias=False)
        self.l1 = nn.Linear(7 * 7 * 4, 10)
    def forward(self, x):
        y = self.c1(x).relu().max_pool()
        y = self.c2(y).relu().max_pool()
        y = self.l1(y.reshape(-1, 7 * 7 * 4))
        return y

if __name__ == '__main__':

    # load datasets
    mnist_train = MNIST_Train(shuffle=True, batchsize=128)
    mnist_test = MNIST_Test(shuffle=False, batchsize=128)
    # create model
    model = CNN()
    optim = AdaBelief(model.parameters(), lr=0.001)

    steps = 1000
    # train model
    losses = []
    for i in (pbar := trange(steps)):
        # get a random data sample
        idx = np.random.randint(0, mnist_train.n, size=128)
        x, y_hat = mnist_train[idx]
        x = x.reshape(-1, 1, 28, 28).detach()
        # predict and compute error
        y = model(x)
        l = cross_entropy(y, y_hat)
        # optimize
        optim.zero_grad()
        l.backward()
        optim.step()
        # progress
        losses.append(l.data.item())
        pbar.set_postfix({'loss': sum(losses[-100:]) / min(100, len(losses))})
        pbar.update(1)

    # evaluate model
    total_hits = 0
    for x, y_hat in mnist_test:
        y = model(x.reshape(x.shape[0], 1, 28, 28))
        y = np.argmax(y.data, axis=1)
        total_hits += (y == y_hat.data).sum()
    acc = total_hits / mnist_test.n
    print("Accuracy:\t", acc, end='\n\n')

    # infer model
    mnist_test.shuffle()
    x, target = mnist_test[0]
    predict = np.argmax(model(x.reshape(1, 1, 28, 28)).data, axis=1)[0]
    print("Prediction:\t", predict, "\nTarget:\t\t", target.item())

    # visualize convolutions
    w = model.c1.w.data
    f, axes = plt.subplots(w.shape[0], w.shape[1])
    axes = axes.reshape(w.shape[0], w.shape[1])
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            axes[i, j].imshow(w[i, j, ...], cmap='gray')
    # plt.show()