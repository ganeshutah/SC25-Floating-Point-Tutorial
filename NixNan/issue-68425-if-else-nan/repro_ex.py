import torch
a = torch.tensor(100., requires_grad=True)
b = torch.where(a <= 0, torch.exp(a), 1 + a)
b.backward()

print(b)
print(a.grad)
