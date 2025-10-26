import torch
import numpy as np

from torch.masked import masked_tensor
from torch.masked import as_masked_tensor


# PyTorch Issue 4132 - when using mask, x/0 yields NaN grad

x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]

mask = (div != 0) # => mask is [0, 1]
loss = y[mask]
loss.backward()

x.grad # grad is [nan, 1], but expected [0, 1]


x = torch.tensor([1., 1.], requires_grad=True)
div = torch.tensor([0., 1.])
y = x/div # => y is [inf, 1]

mask = (div != 0) # => mask is [0, 1]
loss = as_masked_tensor(y, mask)
# We could add autograd support for indexing here instead of using sum
loss = loss.sum()
loss.backward()

x.grad # grad is [nan, 1], but expected [0, 1]



