import torch
import matplotlib.pyplot as plt
from flamecalc import CalVarSolver


def f(y, dy, x):
    return torch.sqrt((1 + dy**2) / (-y + 0.000001))


if __name__ == '__main__':
    A = (0, 0)
    B = (3.14159, -2)
    domain = torch.linspace(0, 3.14159, 20000)

    model = CalVarSolver(f, A, B, domain)

    result = model.optimize(lr=0.003, epoch=100)
    y = result["y"].detach().numpy()[0]
    plt.plot(domain, y)
    plt.show()
