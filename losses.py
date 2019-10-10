import torch


def norm_triangle(x, add_x, x2):
    print('---NORM TRIANGLE---')
    print(x.shape, add_x.shape, x2.shape)
    x_norm = torch.norm(x, p=2, dim=1)
    add_x_norm = torch.norm(add_x, p=2, dim=1)
    x2_norm = torch.norm(x2, p=2, dim=1)
    violations = torch.sub(x_norm, x2_norm+add_x_norm)
    violations[violations < 0] = 0.0
    print(violations.shape)
    print('---')
    return torch.mean(violations)