import torch
from torch.optim import Adam
import matplotlib.pyplot as plt

from flamecalc import BVPSolver
from math import pi


def f(y, dy, x):
    return torch.sqrt((1 + dy**2) / (-y + 0.000001))


if __name__ == '__main__':
    A = (0, 0)
    B = (pi, -2)
    domain = torch.linspace(A[0], B[0], 20000)
    model = BVPSolver(f, A, B, domain)

    initial_theta = [-1.0] + [0] * 20
    theta = torch.tensor([initial_theta], requires_grad=True).float()

    optimizer = Adam([theta], lr=0.002)

    losses = []
    epoch = 300

    for i in range(epoch):
        result = model(theta)
        ys, dys = result["y"], result["dy"]
        loss = result["result"]
        optimizer.zero_grad()
        print(f"Epoch {i +1}/{epoch}: {loss}", end='\r')
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().numpy()[0])
        if i % 50 == 0:
            plt.plot(domain, ys.detach().numpy()[0])
    plt.show()
    plt.plot(domain, ys.detach().numpy()[0])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.plot(range(epoch), losses)
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.show()
