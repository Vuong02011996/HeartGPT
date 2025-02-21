import torch
import torch.nn as nn
from torch.nn import functional as F

def check_cross_entropy():
    # Example of target with class indices
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randint(5, (3,), dtype=torch.int64)
    # loss = F.cross_entropy(input, target)
    # loss.backward()

    # Example of target with class probabilities
    input = torch.randn(3, 5, requires_grad=True)
    # tensor([[-0.6362, -0.4409, -1.0615, -0.5267, -0.6832],
    #         [ 0.0596,  0.6266, -0.3978, -0.0122, -0.4236],
    #         [-1.2951, -1.4328, -0.7155,  1.1644,  1.4542]], requires_grad=True)
    target = torch.randn(3, 5).softmax(dim=1)
    loss = F.cross_entropy(input, target)
    loss.backward()


if __name__ == '__main__':
    check_cross_entropy()