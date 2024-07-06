"""
Factory functions for calculating convolutions of timeseries
with discrete distributions of times-to-event using PyTorch.
Factories generate functions that can be applied iteratively to 
simulate the `scan` operation found in JAX.
"""
import torch
from torch import Tensor
from typing import Callable, Tuple
import torch.nn.functional as F

def new_convolve_scanner(
    array_to_convolve: Tensor,
    transform: Callable[[float], float]
) -> Callable:
    """
    Factory function to create a "scanner" function in PyTorch.
    """
    def _new_scanner(history_subset: Tensor, multiplier: float) -> Tuple[Tensor, float]:
        new_val = transform(multiplier * torch.dot(array_to_convolve, history_subset))
        latest = torch.cat((history_subset[1:], new_val.unsqueeze(0)))
        return latest, new_val.item()

    return _new_scanner

def new_double_convolve_scanner(
    arrays_to_convolve: Tuple[Tensor, Tensor],
    transforms: Tuple[Callable[[float], float], Callable[[float], float]]
) -> Callable:
    """
    Factory function to create a double convolution scanner in PyTorch.
    """
    arr1, arr2 = arrays_to_convolve
    t1, t2 = transforms

    def _new_scanner(
        history_subset: Tensor,
        multipliers: Tuple[float, float]
    ) -> Tuple[Tensor, Tuple[float, float]]:
        m1, m2 = multipliers
        m_net1 = t1(m1 * torch.dot(arr1, history_subset))
        new_val = t2(m2 * m_net1 * torch.dot(arr2, history_subset))
        latest = torch.cat((history_subset[1:], new_val.unsqueeze(0)))
        return latest, (new_val.item(), m_net1)

    return _new_scanner



# """ from __future__ import annotations

# from typing import Callable, Tuple
# import torch
# from torch import Tensor

# import torch
# from torch import Tensor
# import numpy as np

def torch_convolve(a: Tensor, v: Tensor, mode: str = 'full') -> Tensor:
    """
    Perform a one-dimensional convolution of two sequences using PyTorch tensors.

    Parameters:
    -----------
    a : Tensor
        The first input sequence (1D tensor) to convolve.
    v : Tensor
        The second input sequence (1D tensor), commonly known as the kernel.
    mode : str, optional
        A string indicating the size of the output:
        'full' (default): returns the convolution at each point of overlap,
        'valid': returns only those parts of the convolution that are computed without zero-padded edges,
        'same': returns the central part of the convolution that is the same size as the largest tensor.

    Returns:
    --------
    Tensor
        The convolution of tensors `a` and `v`.

    Raises:
    -------
    ValueError
        If an invalid mode is passed.

    Examples:
    ---------
    >>> a = torch.tensor([1, 2, 3], dtype=torch.float32)
    >>> v = torch.tensor([0, 1, 0.5], dtype=torch.float32)
    >>> torch_convolve(a, v, mode='full')
    tensor([0.0, 1.0, 2.5, 1.5, 1.5])

    Notes:
    ------
    This function uses PyTorch's `torch.nn.functional.conv1d` which expects tensors
    to be in the format of (batch, channel, length), thus input tensors
    are reshaped and adjusted accordingly.
    """
    if (len(v) > len(a)):
        a, v = v, a

    # Reshape tensors to fit the (batch, channel, length) format required by torch.nn.functional.conv1d
    a = a.view(1, 1, -1)
    v = v.view(1, 1, -1)

    # Reverse the kernel and swap dimensions for cross-correlation to behave as convolution
    v = torch.flip(v, [2])

    # Determine padding based on the convolution mode
    if mode == 'full':
        padding = v.shape[2] - 1
    elif mode == 'valid':
        padding = 0
    elif mode == 'same':
        kernel_size = v.shape[2]
        # If kernel size is even, split the pad unequally to create a center point. 
        # If kernel size is odd, pad equally on both sides to keep the center point
        padding = (kernel_size //2,  kernel_size //2 - 1 + (kernel_size % 2))
        padded_a = F.pad(a, pad=padding, mode='constant', value=0)
        return F.conv1d(padded_a, v)
    else:
        raise ValueError(f"Invalid mode {mode}. Mode must be 'full', 'valid', or 'same'.")

    # Perform the convolution
    result = F.conv1d(a, v, padding=padding)

    # Remove extra dimensions to get the final 1D tensor
    return result.view(-1)


# def new_convolve_scanner(
#     array_to_convolve: Tensor,
#     transform: Callable[[float], float]
# ) -> Callable:
#     """
#     Factory function to create a "scanner" function that can be used to
#     construct an array via backward-looking iterative convolution.
#     """

#     def _new_scanner(
#         history_subset: Tensor, multiplier: float
#     ) -> Tuple[Tensor, float]:
#         """
#         Applies a single step of backward-looking convolution.
#         """
#         new_val = transform(
#             multiplier * torch.dot(array_to_convolve, history_subset)
#         )
#         latest = torch.cat([history_subset[1:], new_val.unsqueeze(0)])
#         return latest, new_val

#     return _new_scanner


# def new_double_convolve_scanner(
#     arrays_to_convolve: Tuple[Tensor, Tensor],
#     transforms: Tuple[Callable[[float], float], Callable[[float], float]]
# ) -> Callable:
#     """
#     Factory function to create a scanner function that iteratively constructs
#     arrays by applying the dot-product/multiply/transform operation twice per
#     history subset, with the first operation yielding an additional scalar
#     multiplier for the second.
#     """
#     arr1, arr2 = arrays_to_convolve
#     t1, t2 = transforms

#     def _new_scanner(
#         history_subset: Tensor,
#         multipliers: Tuple[float, float]
#     ) -> Tuple[Tensor, Tuple[float, float]]:
#         """
#         Applies two sets of convolution, multiply, and transform operations
#         in sequence to construct a new array by scanning along a pair of input
#         arrays that are equal in length to each other.
#         """
#         m1, m2 = multipliers
#         m_net1 = t1(m1 * torch.dot(arr1, history_subset))
#         new_val = t2(m2 * m_net1 * torch.dot(arr2, history_subset))
#         latest = torch.cat([history_subset[1:], new_val.unsqueeze(0)])
#         return latest, (new_val, m_net1)

#     return _new_scanner
#  """