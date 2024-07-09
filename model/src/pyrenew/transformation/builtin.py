"""
Built-in pyrenew transformations created using `pyro.distributions.transforms`.
"""

import pyro.distributions.transforms as pt
import torch
from torch.distributions.transforms import Transform
from torch.distributions.constraints import Constraint

class IdentityTransform(Transform):
    """
    A transform that computes the identity function.
    """
    def __init__(self):
        super().__init__()

    def __eq__(self, other):
        return isinstance(other, IdentityTransform)

    def __call__(self, x):
        return x

    def _inverse(self, y):
        return y

    def log_abs_det_jacobian(self, x, y):
        # The log absolute determinant of the Jacobian of the identity function is zero
        return torch.zeros_like(x)

    def event_dim(self):
        # This method needs to be defined and specifies the number of dimensions
        # from the right that are transformed.
        return 0

# Constraint class for identity transform, generally identity transform works over real numbers
class IdentityConstraint(Constraint):
    def check(self, value):
        return torch.is_tensor(value)

def ScaledLogitTransform(x_max: float) -> pt.ComposeTransform:
    """
    Scaled logistic transformation from the interval (0, X_max) to the interval
    (-infinity, +infinity).

    Parameters
    ----------
    x_max: float
        Maximum value of the untransformed scale (will be transformed to
        +infinity).

    Returns
    -------
    pt.ComposeTransform
        A composition of the following transformations:
        - pyro.distributions.transforms.AffineTransform(0.0, 1.0/x_max)
        - pyro.distributions.transforms.SigmoidTransform().inv
    """
    return pt.ComposeTransform(
        [pt.AffineTransform(0.0, 1.0 / x_max), pt.SigmoidTransform().inv()]
    )

import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform, SigmoidTransform


    
    