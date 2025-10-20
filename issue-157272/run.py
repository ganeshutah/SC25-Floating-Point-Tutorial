import torch

inputs = torch.tensor([complex(float('inf'), 0)] * 3, dtype=torch.complex128)

print("CPU reciprocal:", torch.reciprocal(inputs))
print("GPU reciprocal:", torch.reciprocal(inputs.cuda()))

num = torch.ones_like(inputs)
print("CPU divide:   ", torch.divide(num, inputs))
print("GPU divide:   ", torch.divide(num.cuda(), inputs.cuda()))

print("-" * 50)
inputs = torch.tensor([complex(float('inf'), 0)] * 4, dtype=torch.complex128)

print("CPU reciprocal:", torch.reciprocal(inputs))
print("GPU reciprocal:", torch.reciprocal(inputs.cuda()))

num = torch.ones_like(inputs)
print("CPU divide:   ", torch.divide(num, inputs))
print("GPU divide:   ", torch.divide(num.cuda(), inputs.cuda()))
