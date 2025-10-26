import torch
x = torch.tensor(float('-inf'))
print(torch.nn.functional.silu(x))  # returns nan
