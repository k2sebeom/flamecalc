import torch
import numpy as np
from typing import Tuple


def y0(p_1: Tuple[int, int], p_2: Tuple[int, int],
       xs: torch.Tensor) -> torch.Tensor:
    """ Returns a tensor of the y-values of line between p_1 and p_2

    :param p_1: Point at which the line starts
    :param p_2: Point at which the line ends
    :param xs: Array of x-values of target line
    """
    xs = xs.numpy()
    a = (p_2[1] - p_1[1])/(p_2[0] - p_1[0])
    b = p_1[1]
    ys = a * (xs - xs[0]) + b
    return torch.tensor(ys, requires_grad=False)


def sin_matrix(m: int, xs: torch.Tensor) -> torch.Tensor:
    """ Returns a tensor for sin terms of fourier series

    :param m: Order of the series
    :param xs: Array of x-values of target matrix
    """
    xs = xs.numpy()
    fourier = []
    L = xs[-1] - xs[0]
    for n in range(1, m + 1):
        fourier.append(np.sin(n * np.pi / L * (xs - xs[0])))
    return torch.tensor(fourier, requires_grad=False)


def cos_matrix(m: int, xs: torch.Tensor) -> torch.Tensor:
    """ Returns a tensor for cos terms of fourier series

    :param m: Order of the series
    :param xs: Array of x-values of target matrix
    """
    xs = xs.numpy()
    fourier = []
    L = xs[-1] - xs[0]
    for n in range(1, m + 1):
        w = n * np.pi / L
        fourier.append(w * np.cos(w * (xs - xs[0])))
    return torch.tensor(fourier, requires_grad=False)
