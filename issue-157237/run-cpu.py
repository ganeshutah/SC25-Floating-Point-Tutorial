import torch

torch.manual_seed(1)

device = torch.device("cpu")

# convolutional layer and padding used to make the operation "causal"
kernel_size = 3
conv = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size).to(
    device
)
pad = torch.nn.ZeroPad1d((kernel_size - 1, 0)).to(device)

# a length-12 sequence with 4 NaN elements at the end
incomplete_sequence = torch.randn(1, 1, 12).to(device)
incomplete_sequence[:, :, -4:] = torch.nan
print(f"Test sequence has {incomplete_sequence.isnan().sum()} input NaNs.\n")

# test the operation for three different batch sizes
for batch_size in [4, 16, 32]:

    # a new batch with random data
    x = torch.randn(batch_size, 1, 12).to(device)

    # insert the incomplete sequence in position 3 (doesn't matter which position)
    x[3] = incomplete_sequence

    # apply padding and convolution that should NOT produce additional NaNs
    with torch.no_grad():
        y = conv(pad(x))

    print(
        f"Batch Size: {batch_size:<3} -> Output NaNs for the sequence: {y[3].isnan().sum()}"
    )
