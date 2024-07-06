import torch
import pyro
import pyro.distributions as dist

import torch
import pyro
import pyro.distributions as dist

class ARProcess:
    """
    Object to represent an AR(p) process in Pyro.
    """
    def __init__(self, mean: float, autoreg: torch.Tensor, noise_sd: float):
        """
        Initialize the ARProcess class with mean, autoregressive coefficients, and noise standard deviation.
        """
        self.mean = mean
        self.autoreg = autoreg
        self.noise_sd = noise_sd

    def sample(self, duration: int, inits: torch.Tensor = None, name: str = "arprocess"):
        """
        Sample from the AR process using Pyro

        Parameters
        ----------
        duration: int
            Length of the sequence.
        inits : torch.Tensor, optional
            Initial points, if None, then these are sampled.
            Defaults to None.
        name : str, optional
            Name of the parameter passed to pyro.sample.
            Defaults to "arprocess".

        Returns
        -------
        torch.Tensor
            A tensor of shape (duration,) containing the sampled AR process.
        """
        order = len(self.autoreg)
        if inits is None:
            # Sample initial values if not provided, centered around the mean
            inits = pyro.sample(name + "_init", dist.Normal(self.mean, self.noise_sd).expand([order]).to_event(1))

        # Prepare a tensor to hold the entire AR process
        ts = torch.zeros(duration)
        ts[:order] = inits

        # Use pyro.plate for vectorized noise sampling
        with pyro.plate(f"{name}_noise_plate", duration - order):
            noise = pyro.sample(f"{name}_noise", dist.Normal(0, self.noise_sd).expand([duration - order]))

        # Generate the AR process values
        for t in range(order, duration):
            # Calculate the next value using the autoregressive formula
            current_state = ts[t-order:t]
            next_value = self.mean + torch.dot(self.autoreg, current_state - self.mean) + noise[t-order]
            ts[t] = next_value

        return ts