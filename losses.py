import torch


def norm_triangle(x, x2, x3, l):
    loss = torch.nn.MSELoss()
    return


def zero_loss(fx):
    return torch.mean(torch.abs(fx))
