import torch
import torch
imag_val = 8.5071e+37
input = torch.zeros(4, dtype=torch.cfloat)
input = input + 1j * imag_val
input_gpu = input.cuda()
out_gpu = torch.fft.ifft(input_gpu)
out_cpu = torch.fft.ifft(input)
print(out_cpu)
print(out_gpu.cpu())

# CPU result
# →tensor([0.+infj, 0.+0.j, 0.+0.j, 0.+0.j])
# GPU result
# →tensor([nan+infj, 0.+0.j, 0.+0.j, 0.+0.j])
