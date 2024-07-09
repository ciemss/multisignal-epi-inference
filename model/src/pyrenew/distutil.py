# -*- coding: utf-8 -*-

"""
distutil

Utilities for working with commonly-
encountered probability distributions
found in renewal equation modeling,
such as discrete time-to-event distributions
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import constraints
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform
import pyro.distributions as dist

def validate_discrete_dist_vector(
    discrete_dist: Tensor, tol: float = 1e-5
) -> Tensor:
    """
    Validate that a vector represents a discrete
    probability distribution to within a specified
    tolerance, raising a ValueError if not.

    Parameters
    ----------
    discrete_dist : Tensor
        A PyTorch tensor containing non-negative values that
        represent a discrete probability distribution. The values
        must sum to 1 within the specified tolerance.
    tol : float, optional
        The tolerance within which the sum of the distribution must
        be 1. Defaults to 1e-5.

    Returns
    -------
    Tensor
        The normalized distribution tensor if the input is valid.

    Raises
    ------
    ValueError
        If any value in discrete_dist is negative or if the sum of the
        distribution does not equal 1 within the specified tolerance.
    """
    discrete_dist = discrete_dist.flatten()
    if not torch.all(discrete_dist >= 0):
        raise ValueError(
            "Discrete distribution "
            "vector must have "
            "only non-negative "
            "entries; got {}"
            "".format(discrete_dist)
        )
    dist_norm = torch.sum(discrete_dist)
    if not torch.abs(dist_norm - 1) < tol:
        raise ValueError(
            "Discrete generation interval "
            "distributions must sum to 1 "
            "with a tolerance of {}"
            "".format(tol)
        )
    return discrete_dist / dist_norm


def reverse_discrete_dist_vector(dist: Tensor) -> Tensor:
    """
    Reverse a discrete distribution
    vector (useful for discrete
    time-to-event distributions).

    Parameters
    ----------
    dist : Tensor
        A discrete distribution vector (likely discrete time-to-event distribution)

    Returns
    -------
    Tensor
        A reversed (torch.flip) discrete distribution vector
    """
    return torch.flip(dist, dims=[0])


