"""
Unit tests for the iterative convolution
scanner function factories found in pyrenew.convolve
"""
import torch
import numpy as np
from numpy.testing import assert_array_equal
import pyrenew.convolve as pc
import pytest

# def test_double_scanner_reduces_to_single():
#     """
#     Test for PyTorch that checks the equivalence of single and double scanners.
#     """
#     inits = torch.tensor([0.352, 5.2, -3], dtype=torch.float32)
#     to_scan_a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

#     multipliers = torch.tensor(np.random.normal(0, 0.5, size=500), dtype=torch.float32)

#     def transform_a(x):
#         return 4 * x + 0.025

#     scanner_a = pc.new_convolve_scanner(to_scan_a, transform_a)
#     double_scanner_a = pc.new_double_convolve_scanner(
#         (torch.tensor([523, 2, -0.5233], dtype=torch.float32), to_scan_a),
#         (lambda x: 1, transform_a)
#     )

#     # Manually replicate scanning
#     history = inits
#     results_a = []
#     for multiplier in multipliers:
#         history, result = scanner_a(history, multiplier)
#         results_a.append(result)

#     history = inits
#     results_a_double = []
#     multipliers_b = multipliers * 0.2352
#     for m1, m2 in zip(multipliers_b, multipliers):
#         history, (result, _) = double_scanner_a(history, (m1, m2))
#         results_a_double.append(result)

#     results_a = torch.tensor(results_a)
#     results_a_double = torch.tensor(results_a_double)

#     assert_array_equal(results_a_double.numpy(), np.ones_like(multipliers.numpy()))
#     assert_array_equal(results_a.numpy(), results_a_double.numpy())

#     import pytest

# Assuming torch_convolve is already imported if not defined here again

def test_full_mode():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    v = torch.tensor([0, 1, 0.5], dtype=torch.float32)
    expected = torch.from_numpy(np.convolve(a, v, mode='full'))
    result = pc.torch_convolve(a, v, mode='full')
    assert torch.allclose(result, expected, atol=1e-05), "Full mode convolution failed"

def test_valid_mode():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    v = torch.tensor([1, 0.5], dtype=torch.float32)
    expected = torch.from_numpy(np.convolve(a, v, mode='valid'))  # Valid convolution does not include the zero-padded edges
    result = pc.torch_convolve(a, v, mode='valid')
    assert torch.allclose(result, expected, atol=1e-05), "Valid mode convolution failed"

def test_same_mode_even_kernel():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    v = torch.tensor([1, 0.5], dtype=torch.float32)
    expected = torch.from_numpy(np.convolve(a, v, mode='same'))  # Same mode output size matches the largest input size
    result = pc.torch_convolve(a, v, mode='same')
    assert torch.allclose(result, expected, atol=1e-05 ), "Same mode convolution with even kernel failed"


def test_same_mode_odd_kernel():
    a = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    v = torch.tensor([1, 0.5, 0.25], dtype=torch.float32)  # Even kernel size
    expected = torch.from_numpy(np.convolve(a, v, mode='same')) 
    result = pc.torch_convolve(a, v, mode='same')
    assert torch.allclose(result, expected, atol=1e-05), "Same mode convolution with odd kernel failed"

def test_invalid_mode():
    a = torch.tensor([1, 2, 3], dtype=torch.float32)
    v = torch.tensor([0, 1, 0.5], dtype=torch.float32)
    with pytest.raises(ValueError):
        pc.torch_convolve(a, v, mode='invalid')

# 