"""
This module exposes Pyro's transformations module to the user,
and defines and adds additional custom transformations
"""

from pyro.distributions.transforms import *  # Import all standard transforms
from pyro.distributions.transforms import (
    __all__ as pyro_public_transforms,  # Import list of all public names in transforms
)
from pyrenew.transformation.builtin import ScaledLogitTransform, IdentityTransform  # Custom transformation

# Extend the publicly available transforms with the custom ones
__all__ = ["ScaledLogitTransform", "IdentityTransform"] + pyro_public_transforms