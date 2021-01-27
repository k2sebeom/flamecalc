from torch import nn
from torch.optim import SGD, Adam
from flamecalc.utils import *
from tqdm import tqdm


class BVPSolver(nn.Module):
    def __init__(self, functional, start_point, end_point, domain=None):
        super(BVPSolver, self).__init__()
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


class IVPSolver(nn.Module):
    def __init__(self, functional, r0, dr0, domain):
        super(IVPSolver, self).__init__()
        self.functional = functional
        self.r_0 = r0
        self.dr_0 = dr0
        self.x_t = None
        self.dx_t = None
        if domain[0] != self.r_0[0]:
            raise ValueError(
                "Given domain and initial conditions do not match"
            )
        self.domain = domain

    def forward(self, theta: torch.Tensor):
        v_0 = v0(self.r_0[1], self.dr_0[1], self.domain)
        m = theta.shape[1]
        if self.x_t is None:
            self.x_t = taylor_matrix(m, self.domain)
        if self.dx_t is None:
            self.dx_t = taylor_matrix_diff(m, self.domain)
        residual = torch.matmul(theta, self.x_t)
        residual_d = torch.matmul(theta, self.dx_t)
        y = v_0 + residual
        dy = self.dr_0[1] + residual_d
        fy = self.functional(y, dy, self.domain.unsqueeze(0))
        result = torch.trapz(fy, self.domain.unsqueeze(0))
        return {'result': result, 'y': y, 'dy': dy}

    def optimize(self, theta=None, optimizer=None, lr=0.002, epoch=20000, m=7):
        if theta is None:
            theta = torch.zeros((1, m), requires_grad=True)
        losses = []
        add_loss = losses.append
        if optimizer is None:
            optimizer = Adam([theta], lr=lr)
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
