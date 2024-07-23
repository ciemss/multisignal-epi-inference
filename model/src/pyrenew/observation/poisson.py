# -*- coding: utf-8 -*-
import pyro
import pyro.distributions as dist
import torch
from torch import Tensor
from pyrenew.metaclass import RandomVariable


class PoissonObservation(RandomVariable):
    """
    Poisson observation process in Pyro
    """

    def __init__(self, parameter_name: str = "poisson_rv", eps: float = 1e-8):
        """
        Initializes the PoissonObservation class with a parameter name and an epsilon value for numerical stability.

        Parameters:
        -----------
        parameter_name : str
            Name to use for the Pyro sample. Defaults to 'poisson_rv'.
        eps : float
            Small value added to the rate parameter to avoid zero rates. Defaults to 1e-8.
        """
        self.parameter_name = parameter_name
        self.eps = eps

    def sample(self, mu: Tensor, obs: Tensor = None, name: str = None, **kwargs) -> tuple:
        """
        Sample from the Poisson distribution using Pyro's primitives.

        Parameters:
        -----------
        mu : Tensor
            Rate parameter of the Poisson distribution.
        obs : Tensor, optional
            Observed data, if any.
        name : str, optional
            Custom name for the sample, defaults to internal parameter name.

        Returns:
        --------
        Sample from the Poisson distribution.
        """
        if name is None:
            name = self.parameter_name

        # Ensure the rate parameter is non-zero for stability
        rate = mu + self.eps

        # Sampling from the Poisson distribution
        return (pyro.sample(name, dist.Poisson(rate), obs=obs),)

    @staticmethod
    def validate():
        """ Additional validation steps can be implemented here """
        pass