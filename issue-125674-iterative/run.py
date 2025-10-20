
import torch
import torch.nn as nn
import torch.optim as optim

embed_dim = 128 # !!! change this to 64 and the error will occur only after thousands of iterations !!!
batch_size = 512
seq_length = 16
nhead = 8
num_iterations = 100000
learning_rate = 0.01
device = torch.device("cuda")
torch.autograd.set_detect_anomaly(True)

# Initialize model
model = nn.TransformerEncoderLayer(embed_dim, nhead=nhead, dropout=0.0, batch_first=True, dim_feedforward=1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
inputs = torch.full((batch_size, seq_length, embed_dim), 1000.0, device=device)

# Training loop
for i in range(num_iterations):
    print(f'Iteration {i + 1}')
    optimizer.zero_grad()
    output = model(inputs)
    loss = output.mean()
    loss.backward()
    optimizer.step()

