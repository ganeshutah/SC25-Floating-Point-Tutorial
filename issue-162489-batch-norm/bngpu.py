DEVICE = 'cuda'
device = DEVICE
import torch
from torch.nn import BatchNorm1d, LayerNorm
torch.manual_seed(42)
torch.cuda.manual_seed(42)
input_tensor = torch.linspace(-1e+30, 1e+30, steps=4).reshape(2, 2).to(
    device)
norm_layer = BatchNorm1d(2).to(device)

output_bn = torch.alias_copy(norm_layer(input_tensor))

print(output_bn)
