import torch

def test_silu_gpu():
    if torch.cuda.is_available():
        x = torch.tensor(float('-inf')).cuda()
        print(f"Input: {x}")
        result = torch.nn.functional.silu(x)
        print(f"SiLU output: {result}")
        return result
    else:
        print("CUDA not available")
        return None

if __name__ == "__main__":
    test_silu_gpu()

# LD_PRELOAD=~/nixnan.so python silu_nan_test.py

