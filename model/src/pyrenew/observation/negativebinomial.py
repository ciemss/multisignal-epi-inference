# -*- coding: utf-8 -*-
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import NegativeBinomial
from pyrenew.metaclass import RandomVariable

class NegativeBinomialObservation(RandomVariable):
    """Negative Binomial observation in Pyro"""

    def __init__(self, concentration_prior, concentration_suffix="_concentration",
                 parameter_name="negbinom_rv", eps=1e-10):
        """
        Initialize the NegativeBinomialObservation class with concentration prior,
        a suffix for naming, and epsilon for numerical stability.
        """
        self.validate(concentration_prior)

        self.concentration_prior = concentration_prior
        self.concentration_suffix = concentration_suffix
        self.parameter_name = parameter_name
        self.eps = eps

    def sample(self, mu, obs=None, name=None, **kwargs):
        """
        Sample from the negative binomial distribution using Pyro's primitives.

        Parameters:
        -----------
        mu : torch.Tensor
            Mean parameter of the negative binomial distribution.
        obs : torch.Tensor, optional
            Observed data, if any.
        name : str, optional
            Custom name for the sample, defaults to internal parameter name.

        Returns:
        --------
        Sample from the Negative Binomial distribution.
        """
        if name is None:
            name = self.parameter_name

        concentration = pyro.sample(
            self.parameter_name + self.concentration_suffix,
            self.concentration_prior
        )

        negative_binomial = NegativeBinomial(total_count=concentration,
                                             probs=(mu + self.eps) / (mu + self.eps + concentration))

        return pyro.sample(name, negative_binomial, obs=obs)

    @staticmethod
    def validate(concentration_prior):
        """
        Validate that concentration_prior is a proper distribution or a number.

        Parameters:
        -----------
        concentration_prior : dist.Distribution or Number
            The prior distribution for the concentration parameter of the negative binomial distribution.
        """
        assert isinstance(concentration_prior, (dist.Distribution, torch.Tensor, float, int)), \
            "concentration_prior must be either a Pyro distribution or a numeric value."