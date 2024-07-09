"""
Factory functions for calculating convolutions of timeseries
with discrete distributions of times-to-event using PyTorch.
Factories generate functions that can be applied iteratively to 
simulate the `scan` operation found in JAX.
"""
import torch
from torch import Tensor
from typing import Callable, Tuple, Union
import torch.nn.functional as F

# def new_convolve_scanner(
#     array_to_convolve: Tensor,
#     transform: Callable[[float], float]
# ) -> Callable:
#     """
#     Factory function to create a "scanner" function in PyTorch.
#     """
#     def _new_scanner(history_subset: Tensor, multiplier: float) -> Tuple[Tensor, float]:
#         new_val = transform(multiplier * torch.dot(array_to_convolve, history_subset))
#         latest = torch.cat((history_subset[1:], new_val.unsqueeze(0)))
#         return latest, new_val.item()

#     return _new_scanner

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



def torch_convolve_scanner(
    array_to_convolve: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor]
) -> Callable:
    """
    Creates a "scanner" function for use with torch to construct an array via
    backward-looking iterative convolution.

    Parameters:
    ----------
    array_to_convolve : torch.Tensor
        A 1D tensor to convolve with subsets of the iteratively constructed history array.

    transform : Callable
        A transformation to apply to the result of the dot product and multiplication.

    Returns:
    -------
    Callable
        A scanner function that can be used for convolution.

    Notes:
    -----
    Implements the convolution:
        X(t) = f(m(t) * (X[t-n:t] * d))
    where 'd' is array_to_convolve and 'f' is transform.
    """
    def _scanner(history_subset: torch.Tensor, multipliers: Union[Tuple[torch.Tensor, ...], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single scan operation by applying a transformation to the dot product of a given array with a history subset,
        followed by concatenating the result to update the history for the next step.

        This scanner function is designed to handle inputs either as a tuple of multipliers or a single tensor multiplier. This flexibility
        allows the function to be used in scenarios where different transformations might be applied based on the input structure.

        Parameters:
        ----------
        history_subset : torch.Tensor
            A tensor representing the current state of the history that is being updated iteratively. This subset is typically
            the slice of a larger array that represents past values that influence the next value.

        multipliers : Union[Tuple[torch.Tensor, ...], torch.Tensor]
            The multipliers to apply to the dot product of the `array_to_convolve` with `history_subset`. If a tuple is provided,
            it assumes a structure where each element of the tuple influences the computation distinctly, allowing for complex
            behaviors like feedback mechanisms or multi-stage transformations.

        Returns:
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The updated history subset tensor after adding the new computed value.
            - The new value tensor after applying the dot product and transformation, used for further computations or as part of an output sequence.

        Raises:
        ------
        RuntimeError:
            If there is a dimension mismatch in tensor operations, indicating that the expected input shapes are not aligned.

        Examples:
        --------
        Assuming a setup with `array_to_convolve` and a transformation function `transform` defined:

        >>> history = torch.tensor([1.0, 2.0, 3.0])
        >>> multiplier = torch.tensor(0.5)
        >>> updated_history, new_value = _scanner(history, multiplier)
        """
        if isinstance(multipliers, tuple):
            # Unpack tuple if multipliers are given as a tuple
            m1, *_ = multipliers
            multiplier_effect = m1 * torch.dot(array_to_convolve, history_subset)
        else:
            # Use the multiplier directly if it's a single tensor
            multiplier_effect = multipliers * torch.dot(array_to_convolve, history_subset)

        new_val = transform(multiplier_effect)
        if new_val.dim() == 0:
            new_val = new_val.unsqueeze(0)

        latest = torch.cat((history_subset[1:], new_val), dim=0)
        return latest, new_val
    return _scanner

def torch_double_convolve_scanner(
    arrays_to_convolve: Tuple[torch.Tensor, torch.Tensor],
    transforms: Tuple[Callable, Callable]
) -> Callable:
    """
    Creates a scanner function that applies two sets of convolution operations in sequence.

    Parameters:
    ----------
    arrays_to_convolve : Tuple[torch.Tensor, torch.Tensor]
        Two tensors for convolution operations.

    transforms : Tuple[Callable, Callable]
        Two functions, each transforming the output of the convolution operations.

    Returns:
    -------
    Callable
        A scanner function for sequential convolutions.

    Notes:
    -----
    Applies two convolution operations in sequence, each followed by a transformation.
    """
    arr1, arr2 = arrays_to_convolve
    t1, t2 = transforms

    def _scanner(history_subset: torch.Tensor, multipliers: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[float, float]]:
        m1, m2 = multipliers
        # Assuming arr1 and arr2 are already defined and accessible in this context
        m_net1 = t1(m1 * torch.dot(arr1, history_subset))
        new_val = t2(m2 * m_net1 * torch.dot(arr2, history_subset))
        
        # If new_val is a scalar, convert it to a 1D tensor by adding a new axis
        if new_val.dim() == 0:
            new_val = new_val.unsqueeze(0)
        # Now both history_subset[1:] and new_val are 1D
        latest = torch.cat((history_subset[1:], new_val), dim=0)  # Concatenate along the first dimension
        return latest, (new_val.item(), m_net1)

    return _scanner



def torch_scan(f, init, xs):
    """
    A version of scan for PyTorch that handles functions returning tuples of tensors.

    Parameters:
    -----------
    f : Callable
        Function to apply at each step, takes (current_state, current_inputs) and returns new_state, result(s).
    init : Tensor or tuple
        Initial state to start the scan from.
    xs : Tensor or tuple of Tensors
        Sequence or sequences of inputs to apply 'f' over.

    Returns:
    --------
    final_state : Tensor or tuple
        The final state after applying 'f' across all elements.
    results : tuple of Tensors
        Collected results from each step of applying 'f', each element in the tuple corresponds to one part of the tuple returned by 'f'.
    """
    current = init
    outputs = []
    for x in xs:
        if isinstance(x, (list, tuple)):
            current, out = f(current, x)
        else:
            current, out = f(current, (x,))  # Ensure it's passed as a tuple
        outputs.append(out)
    return torch.stack(outputs), current

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