# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import torch

class InfectionSeedMethod(metaclass=ABCMeta):
    """Method for seeding initial infections in a renewal process."""

    def __init__(self, n_timepoints: int):
        """
        Default constructor for the InfectionSeedMethod class.
        Parameters
        ----------
        n_timepoints : int
            the number of time points to generate seed infections for
        """
        self.validate(n_timepoints)
        self.n_timepoints = n_timepoints

    @staticmethod
    def validate(n_timepoints: int) -> None:
        """
        Validate inputs for the InfectionSeedMethod class constructor
        Parameters
        ----------
        n_timepoints : int
            the number of time points to generate seed infections for
        """
        if not isinstance(n_timepoints, int):
            raise TypeError(f"n_timepoints must be an integer. Got {type(n_timepoints)}")
        if n_timepoints <= 0:
            raise ValueError(f"n_timepoints must be positive. Got {n_timepoints}")

    @abstractmethod
    def seed_infections(self, I_pre_seed: torch.Tensor):
        """
        Generate the number of seeded infections at each time point.
        Parameters
        ----------
        I_pre_seed : torch.Tensor
            An array representing some number of latent infections to be used with the specified InfectionSeedMethod.
        """

    def __call__(self, I_pre_seed: torch.Tensor):
        return self.seed_infections(I_pre_seed)

class SeedInfectionsZeroPad(InfectionSeedMethod):
    """
    Create a seed infection vector of specified length by
    padding a shorter vector with an appropriate number of
    zeros at the beginning of the time series.
    """

    def seed_infections(self, I_pre_seed: torch.Tensor):
        """
        Pad the seed infections with zeros at the beginning of the time series.
        Parameters
        ----------
        I_pre_seed : torch.Tensor
            An array with seeded infections to be padded with zeros.
        """
        if self.n_timepoints < I_pre_seed.size(0):
            raise ValueError(
                "I_pre_seed must be no longer than n_timepoints. "
                f"Got I_pre_seed of size {I_pre_seed.size(0)} and "
                f"n_timepoints of size {self.n_timepoints}."
            )
        return torch.cat([torch.zeros(self.n_timepoints - I_pre_seed.size(0)), I_pre_seed])

class SeedInfectionsFromVec(InfectionSeedMethod):
    """Create seed infections from a vector of infections."""

    def seed_infections(self, I_pre_seed: torch.Tensor):
        """
        Create seed infections from a vector of infections.
        Parameters
        ----------
        I_pre_seed : torch.Tensor
            An array with the same length as n_timepoints to be used as the seed infections.
        """
        if I_pre_seed.size(0) != self.n_timepoints:
            raise ValueError(
                "I_pre_seed must have the same size as n_timepoints. "
                f"Got I_pre_seed of size {I_pre_seed.size(0)} "
                f"and n_timepoints of size {self.n_timepoints}."
            )
        return I_pre_seed

class SeedInfectionsExponential(InfectionSeedMethod):
    """Generate seed infections according to exponential growth."""

    def __init__(self, n_timepoints: int, rate: float, t_pre_seed: int = None):
        """
        Default constructor for the SeedInfectionsExponential class.
        Parameters
        ----------
        n_timepoints : int
            the number of time points to generate seed infections for
        rate : float
            the rate of exponential growth
        t_pre_seed : int, optional
            The time point whose number of infections is described by I_pre_seed. Defaults to n_timepoints - 1.
        """
        super().__init__(n_timepoints)
        self.rate = rate
        if t_pre_seed is None:
            t_pre_seed = n_timepoints - 1
        self.t_pre_seed = t_pre_seed

    def seed_infections(self, I_pre_seed: torch.Tensor):
        """
        Generate seed infections according to exponential growth.
        Parameters
        ----------
        I_pre_seed : torch.Tensor
            An array of size 1 representing the number of infections at time t_pre_seed.
        """
        if I_pre_seed.numel() != 1:
            raise ValueError(f"I_pre_seed must be a tensor of size 1. Got size {I_pre_seed.numel()}.")
        (rate,) = self.rate.sample()
        if rate.size(0) != 1:
            raise ValueError(f"rate must be an array of size 1. Got size {rate.size}.")
        t = torch.arange(self.n_timepoints) - self.t_pre_seed
        return I_pre_seed * torch.exp(rate * t)
    