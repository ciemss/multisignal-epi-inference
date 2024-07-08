"""
Unit tests for the iterative convolution
scanner function factories found in pyrenew.convolve
"""
import torch
import numpy as np
from numpy.testing import assert_array_equal
import pyrenew.convolve as pc
import pytest
import torch
from torch.testing import assert_allclose

"""
Unit tests for the iterative convolution
scanner function factories found in pyrenew.convolve
"""

import torch
import numpy as np
from numpy.testing import assert_array_equal
import pyrenew.convolve as pc


def test_double_scanner_reduces_to_single():
    """
    Test that torch_double_convolve_scanner() yields a function
    that is equivalent to a single scanner if the first
    scan is chosen appropriately
    """
    inits = torch.tensor([0.352, 5.2, -3], dtype=torch.float32)
    to_scan_a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)

    multipliers = torch.tensor(np.random.normal(0, 0.5, size=500), dtype=torch.float32)

    def transform_a(x):
        """
        Transformation associated with array to_scan_a

        Parameters:
        -----------
        x: float
            Input value

        Returns:
        --------
        The result of 4 * x + 0.025, where x is the input value
        """
        return 4 * x + 0.025

    scanner_a = pc.torch_convolve_scanner(to_scan_a, transform_a)

    double_scanner_a = pc.torch_double_convolve_scanner(
        (torch.tensor([523, 2, -0.5233], dtype=torch.float32), to_scan_a),
        (lambda x: torch.tensor(1.), transform_a)
    )

    _, result_a = pc.torch_scan(scanner_a, inits, multipliers)

    _, result_a_double = pc.torch_scan(
        double_scanner_a, inits, (multipliers * 0.2352, multipliers)
    )

    # Using numpy's assert_array_equal to check equivalence of results
    assert_array_equal(result_a_double[1].numpy(), np.ones_like(multipliers))
    assert_array_equal(result_a_double[0].numpy(), result_a.numpy())


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