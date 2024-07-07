import torch
from pyro.distributions import Normal
import pyro
from pyro.infer import Predictive

class FirstDifferenceARProcess:
    """
    Class for a stochastic process with an AR(1) process on the first differences (i.e., the rate of change).
    """

    def __init__(self, autoreg: torch.Tensor, noise_sd: float):
        """
        Initialize the FirstDifferenceARProcess class.

        Parameters:
        -----------
        autoreg : torch.Tensor
            Coefficient for the autoregressive process.
        noise_sd : float
            Standard deviation of the noise in the AR process.
        """
        self.autoreg = autoreg
        self.noise_sd = noise_sd

    def sample(self, duration: int, init_val: torch.Tensor = None, init_rate_of_change: torch.Tensor = None):
        """
        Sample from the AR(1) process on the first differences.

        Parameters:
        -----------
        duration : int
            Number of time steps to sample.
        init_val : torch.Tensor, optional
            Initial value of the process.
        init_rate_of_change : torch.Tensor, optional
            Initial rate of change of the process.

        Returns:
        --------
        torch.Tensor
            Samples from the AR process.
        """
        if init_rate_of_change is None:
            init_rate_of_change = torch.tensor(0.0)

        if init_val is None:
            init_val = torch.tensor(0.0)

        samples = torch.zeros(duration)
        samples[0] = init_val + init_rate_of_change

        for t in range(1, duration):
            rate_of_change = pyro.sample(f"rate_of_change_{t}", Normal(samples[t-1] * self.autoreg, self.noise_sd))
            samples[t] = samples[t-1] + rate_of_change

        return samples
