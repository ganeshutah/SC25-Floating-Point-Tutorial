import numpy as np
import torch

input = torch.tensor([np.inf, np.inf], dtype=torch.float32)

out = torch.std_mean(input, None)
print(out[1])  # tensor(nan) actual, inf expected.

out = torch.mean(input)
print(out)  # tensor(inf)

"""
torch.std_mean seems to return  if the input array
contains also normal number.
"""

input = torch.tensor([1.0, np.inf], dtype=torch.float32)
out = torch.std_mean(input)
print(out[1])  # tensor(inf)
