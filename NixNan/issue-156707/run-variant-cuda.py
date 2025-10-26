import torch
import torch.nn.functional as F

# Minimal reproduction
q = torch.randn(1, 1, 2, 4)
k = torch.randn(1, 1, 2, 4)
v = torch.randn(1, 1, 2, 4)
mask = torch.tensor([[[[False, False], [True, True]]]], dtype=torch.bool)

# CPU returns zeros
out_cpu = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
print("CPU:", out_cpu[0, 0, 0, :3].tolist())  # [0.0, 0.0, 0.0]

# GPU returns ??
out_cuda = F.scaled_dot_product_attention(q.to('cuda'), k.to('cuda'), v.to('cuda'), attn_mask=mask.to('cuda'))
print("CUDA:", out_cuda[0, 0, 0, :3].tolist())  # [nan, nan, nan]
