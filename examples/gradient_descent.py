import sys
sys.path.insert(0, "../")
import lightgrad as light
import matplotlib.pyplot as plt

# create tensors
a = light.uniform(-1, 1, shape=(10, 10))
b = light.uniform(-1, 1, shape=(10, 10))
c = light.uniform(-1, 1, shape=(10, 10))
# objective to minimize
f = lambda: (a.tanh() + b.sigmoid()) @ (c.relu() - a.sigmoid())

ys = []
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
    y.zero_grad(traverse_graph=True)
    # store sum in array
    ys.append(y.sum().item())

plt.plot(ys)
plt.show()
