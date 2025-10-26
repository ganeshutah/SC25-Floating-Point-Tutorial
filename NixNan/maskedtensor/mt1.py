import torch
import numpy as np

from torch.masked import masked_tensor
from torch.masked import as_masked_tensor


# PyTorch Issue 10729 - torch.where
# This behavior underlies the fix to clamp, which uses where in its derivative
x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
y = torch.where(x < 0, torch.exp(x), torch.ones_like(x))
print("y:", y)
y.sum().backward()
print("x.grad:", x.grad)
print("y.grad:", y.grad)



x = torch.tensor([-10., -5, 0, 5, 10, 50, 60, 70, 80, 90, 100], requires_grad=True)
mask = x < 0
mx = masked_tensor(x, mask, requires_grad=True)
my = masked_tensor(torch.ones_like(x), ~mask, requires_grad=True)
y = torch.where(mask, torch.exp(mx), my)
s = y.sum()
s.backward()
# Gradient is only provided to selected subset.
# Effectively this changes the gradient of where to mask out elements instead
# of setting them to zero.
print("mx.grad: ", mx.grad)


