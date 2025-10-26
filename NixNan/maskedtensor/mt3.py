import torch
import numpy as np

from torch.masked import masked_tensor
from torch.masked import as_masked_tensor

# PyTorch Issue 67180 - torch.nansum and torch.nanmean

a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
c = a * b
c1 = torch.nansum(c)  # or torch.nanmean

bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1


a = torch.tensor([1., 2., float('nan')])
b = torch.tensor(1.0, requires_grad=True)
ma = masked_tensor(a, ~torch.isnan(a))
c = ma * b
c1 = torch.sum(c)  # or torch.nanmean

bgrad1, = torch.autograd.grad(c1, b, retain_graph=True)
bgrad1


