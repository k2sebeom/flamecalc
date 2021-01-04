from torch import nn
from torch.optim import SGD
from flamecalc.utils import *
from tqdm import tqdm


class CalVarSolver(nn.Module):
    def __init__(self, functional, start_point, end_point, domain=None):
        super().__init__()
        self.functional = functional
        self.p_1 = start_point
        self.p_2 = end_point
        if domain is None:
            self.domain = torch.linspace(self.p_1[0], self.p_2[0], 100)
        else:
            if domain[0] != start_point[0] or domain[-1] != end_point[0]:
                raise ValueError(
                    "Given domain and boundary points do not match"
                )
            self.domain = domain

    def forward(self, theta: torch.Tensor):
        m = theta.shape[1]
        y_0 = y0(self.p_1, self.p_2, self.domain)
        x_s = sin_matrix(m, self.domain)
        x_c = cos_matrix(m, self.domain)
        residual = torch.matmul(theta, x_s)
        residual_d = torch.matmul(theta, x_c)
        y = y_0 + residual
        a = (self.p_1[1] - self.p_2[1])/(self.p_1[0] - self.p_2[0])
        dy = a + residual_d
        fy = self.functional(y, dy, self.domain.unsqueeze(0))
        result = torch.trapz(fy, self.domain.unsqueeze(0))
        return {'result': result, 'y': y, 'dy': dy}

    def optimize(self, theta=None, optimizer=None, lr=0.002, epoch=2000, m=5):
        if theta is None:
            theta = torch.rand((1, m), requires_grad=True)
        losses = []
        add_loss = losses.append
        if optimizer is None:
            optimizer = SGD([theta], lr=lr)
        p_bar = tqdm(range(epoch), ncols=150)
        for i in p_bar:
            result = self.forward(theta)
            loss = result["result"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            add_loss(loss.detach().numpy()[0])
            p_bar.set_description(f"Epoch {i + 1}/{epoch} | "
                                  f"value: {loss.detach().numpy()[0]:.5f}")
        result = self.forward(theta)
        y, dy = result["y"], result["dy"]
        return {"theta": theta, "y": y, "dy": dy, "log": losses}
