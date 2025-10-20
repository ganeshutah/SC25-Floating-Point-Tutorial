#!/usr/bin/env python3
import torch

def main():
    # 1. Select device: CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 2. Create tensor 'a' on the chosen device
    a = torch.tensor(2.0, requires_grad=True, device=device)

    # 3. Compute b = where(a <= 0, exp(a), 1 + a)
    b = torch.where(a <= 0, torch.exp(a), 1 + a)

    # 4. Backpropagate
    b.backward()

    # 5. Print results and device placements
    print(f"b = {b.item()}  (device: {b.device})")
    print(f"a.grad = {a.grad.item()}  (device: {a.grad.device})")

    # 6. Sanity check
    assert b.device == a.device == a.grad.device, "Tensors are not all on the same device!"

if __name__ == "__main__":
    main()
