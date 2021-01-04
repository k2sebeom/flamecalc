import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from tqdm import tqdm
from flamecalc import CalVarSolver


def f(y, dy, x):
    return dy * (1 + x**2 * dy)


if __name__ == '__main__':
    A = (1, 1)
    B = (2, 0.5)
    domain = torch.linspace(1, 2, 100)

    model = CalVarSolver(f, A, B, domain)

    theta = torch.rand((1, 5), requires_grad=True)
    optimizer = SGD([theta], lr=0.000002)
    losses = []
    epoch = 5000
    for i in tqdm(range(epoch), ncols=50):
        result = model(theta)
        ys, dys = result["y"], result["dy"]
        loss = result["result"]
        optimizer.zero_grad()
        # print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy()[0])
    plt.plot(domain, ys.detach().numpy()[0])
    plt.show()
    plt.plot(range(epoch), losses)
    plt.show()
    print(theta)
