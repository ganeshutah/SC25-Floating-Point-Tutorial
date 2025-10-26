import torch
import numpy as np

from torch.masked import masked_tensor
from torch.masked import as_masked_tensor

# PyTorch Issue 52248 - another torch.where

# A more recent incarnation specific to where of this
# https://github.com/pytorch/pytorch/issues/52248

a = torch.randn((), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())

print(torch.where(b, a/0, c))
print(torch.autograd.grad(torch.where(b, a/0, c), a))

a = masked_tensor(torch.randn(()), torch.tensor(True), requires_grad=True)
b = torch.tensor(False)
c = torch.ones(())

print(torch.where(b, a/0, c))
print(torch.autograd.grad(torch.where(b, a/0, c), a))
