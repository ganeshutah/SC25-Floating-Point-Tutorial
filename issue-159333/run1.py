import torch
a = torch.tensor([32763, 32764, 32765, 32766, 32767], device='cuda').view(torch.bfloat16)
print(f'{a=}')

b = torch.zeros_like(a)
torch._foreach_copy_([b], [a])
print(f'{b=}')
#print(f'{a.view(torch.uint16)=}')
#print(f'{b.view(torch.uint16)=}')
#(a.view(torch.uint16) == b.view(torch.uint16)).all()
