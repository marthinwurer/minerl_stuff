"""
https://github.com/jiaweizzhao/ZerO-initialization/issues/1#issuecomment-1405598940
"""

import torch
from torch import nn, optim
import math
from scipy.linalg import hadamard

# def hadamard(rank, dtype=None):
#     H = torch.tensor([[1.]], dtype=dtype)
#     for i in range(0, rank): 
#         H = torch.cat([torch.cat([H, H]), torch.cat([H, -H])], dim=1)
#     return H

def init_ZerO_linear(matrix_tensor):
    with torch.no_grad(): 
        m = matrix_tensor.size(0)
        n = matrix_tensor.size(1)

        if m <= n:
            init_matrix = torch.nn.init.eye_(torch.empty(m, n))
        elif m > n:
            clog_m = math.ceil(math.log2(m))
            p = 2**(clog_m)
            init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))

        return nn.Parameter(init_matrix)

def init_ZerO_convolution(convolution_tensor):
    with torch.no_grad():
        m = convolution_tensor.size(0)
        n = convolution_tensor.size(1)
        k = convolution_tensor.size(2)
        l = convolution_tensor.size(3)

        index = int(math.floor(k/2))
        if m <= n:
            convolution_tensor[:, :, index, index] = torch.nn.init.eye_(torch.empty(m, n))

        elif m > n:
            clog_m = math.ceil(math.log2(m))
            p = 2**(clog_m)
            convolution_tensor[:, :, index, index] = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))

        return nn.Parameter(convolution_tensor)

def init_ZerO(module):
    # linear and conv1d weight initialization
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
        module.weight = init_ZerO_linear(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    # conv initialization
    elif isinstance(module, nn.Conv2d):
        module.weight = init_ZerO_convolution(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    # batchnorm initialization: "For ResNet with batch normalization, we follow the standard practice to initialize the scale and bias in batch normalization as one and zero"
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

if __name__ == "__main__":
    net = nn.Sequential(
        nn.Conv2d(2, 8, 3),
        nn.Conv2d(8, 2, 3),
        nn.Linear(2, 8),
        nn.Linear(8, 2))
    net.apply(init_ZerO)
