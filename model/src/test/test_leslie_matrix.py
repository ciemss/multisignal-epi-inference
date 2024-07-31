import torch
import pyrenew.math as pmath

def test_get_leslie():
    """
    Test that get_leslie_matrix returns expected Leslie matrices
    """

    gi = torch.tensor([0.4, 0.2, 0.2, 0.1, 0.1], dtype=torch.float32)
    R_a = 0.4
    R_b = 3.0
    expected_a = torch.tensor(
        [
            [0.16, 0.08, 0.08, 0.04, 0.04],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32
    )
    expected_b = torch.tensor(
        [
            [1.2, 0.6, 0.6, 0.3, 0.3],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32
    )

    assert torch.allclose(pmath.get_leslie_matrix(R_a, gi), expected_a, atol=1e-5), "Leslie matrix for R_a did not match expected"
    assert torch.allclose(pmath.get_leslie_matrix(R_b, gi), expected_b, atol=1e-5), "Leslie matrix for R_b did not match expected"

# Note: This assumes that pmath.get_leslie_matrix() is already adapted to use PyTorch.