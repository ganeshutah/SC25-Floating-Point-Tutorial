import pytest, torch

@pytest.mark.cuda
def test_softmax_781_double_cuda():
    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    c, n = 3, 781
    x = torch.zeros(c, n, dtype=torch.float64, device=device)
    y = torch.softmax(x, dim=1)

    # Expected: each row is uniform 1/n
    target = torch.full((c, n), 1.0/n, dtype=torch.float64, device=device)

    # If the bug is present, y won't match target.
    assert not torch.allclose(y, target, atol=1e-12, rtol=0), (
        "Bug not reproduced on this build (maybe already fixed?)."
    )

