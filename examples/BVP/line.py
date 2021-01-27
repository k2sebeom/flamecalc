import torch
import matplotlib.pyplot as plt
from flamecalc import BVPSolver


def f(y, dy, x):
    return torch.sqrt(1 + dy**2)


if __name__ == '__main__':
    A = (1, 1)
    B = (2, 5)
    domain = torch.linspace(A[0], B[0], 100)

    model = BVPSolver(f, A, B, domain)
    epoch = 2500
    result = model.optimize(lr=0.02, epoch=epoch)

    y = result["y"].detach()[0]
    losses = result["log"]
    plt.plot(domain, y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.plot(range(epoch), losses)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.show()
