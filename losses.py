import torch


def norm_triangle(x, add_x, x2):
    violations = torch.sub(add_x-x2, x)
    violations[violations < 0] = 0.000001
    return torch.mean(violations)


def zero_loss(fx):
    if torch.abs(fx - 0.0) < 0.00001 :
        return 0.0
    else:
        return torch.abs(fx)


def pos_loss(x):
    violations = x
    violations[x > 0] = 0.000001
    return -torch.mean(violations)