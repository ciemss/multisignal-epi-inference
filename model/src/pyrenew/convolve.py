"""
Factory functions for calculating convolutions of timeseries
with discrete distributions of times-to-event using PyTorch.
Factories generate functions that can be applied iteratively to 
simulate the `scan` operation found in JAX.
"""
from __future__ import annotations

from typing import Callable, Tuple
import torch
from torch import Tensor


def new_convolve_scanner(
    array_to_convolve: Tensor,
    transform: Callable[[float], float]
) -> Callable:
    """
    Factory function to create a "scanner" function that can be used to
    construct an array via backward-looking iterative convolution.
    """

    def _new_scanner(
        history_subset: Tensor, multiplier: float
    ) -> Tuple[Tensor, float]:
        """
        Applies a single step of backward-looking convolution.
        """
        new_val = transform(
            multiplier * torch.dot(array_to_convolve, history_subset)
        )
        latest = torch.cat([history_subset[1:], new_val.unsqueeze(0)])
        return latest, new_val

    return _new_scanner


def new_double_convolve_scanner(
    arrays_to_convolve: Tuple[Tensor, Tensor],
    transforms: Tuple[Callable[[float], float], Callable[[float], float]]
) -> Callable:
    """
    Factory function to create a scanner function that iteratively constructs
    arrays by applying the dot-product/multiply/transform operation twice per
    history subset, with the first operation yielding an additional scalar
    multiplier for the second.
    """
    arr1, arr2 = arrays_to_convolve
    t1, t2 = transforms

    def _new_scanner(
        history_subset: Tensor,
        multipliers: Tuple[float, float]
    ) -> Tuple[Tensor, Tuple[float, float]]:
        """
        Applies two sets of convolution, multiply, and transform operations
        in sequence to construct a new array by scanning along a pair of input
        arrays that are equal in length to each other.
        """
        m1, m2 = multipliers
        m_net1 = t1(m1 * torch.dot(arr1, history_subset))
        new_val = t2(m2 * m_net1 * torch.dot(arr2, history_subset))
        latest = torch.cat([history_subset[1:], new_val.unsqueeze(0)])
        return latest, (new_val, m_net1)

    return _new_scanner
