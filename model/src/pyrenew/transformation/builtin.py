"""
Built-in pyrenew transformations created using `pyro.distributions.transforms`.
"""

import pyro.distributions.transforms as pt

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

def IdentityTransform() -> pt.ComposeTransform:
    """
    Identity transformation

    Returns
    -------
    pt.ComposeTransform 
      An empty transformation
    """
    return pt.ComposeTransform([])
    