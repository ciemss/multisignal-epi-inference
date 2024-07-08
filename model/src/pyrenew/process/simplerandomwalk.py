import torch
import pyro
import pyro.distributions as dist
from pyrenew.metaclass import RandomVariable

class SimpleRandomWalkProcess(RandomVariable):
    """
    Class for a Markovian random walk with an arbitrary step distribution
    """

    def __init__(self, error_distribution: dist.Distribution):
        """
        Default constructor

        Parameters
        ----------
        error_distribution : dist.Distribution
            Distribution object used to generate steps in the random walk.

        Returns
        -------
        None
        """
        self.error_distribution = error_distribution

    def sample(self, n_timepoints: int, name: str = "randomwalk", init: float = None, **kwargs):
        """
        Samples from the random walk process.

        Parameters
        ----------
        n_timepoints : int
            Length of the random walk to generate.
        name : str, optional
            Base name for Pyro sample sites, default is "randomwalk".
        init : float, optional
            Initial value of the random walk, default is sampled from the error distribution.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal Pyro sample calls.

        Returns
        -------
        tuple
            A single tensor representing the random walk, with shape (n_timepoints,).
        """
        if init is None:
            init = pyro.sample(name + "_init", self.error_distribution)
        diffs = pyro.sample(
            name + "_diffs",
            self.error_distribution.expand([n_timepoints - 1]).to_event(1)
        )

        return (torch.cat([torch.tensor([init]), init + torch.cumsum(diffs, dim=0)]),)

    @staticmethod
    def validate():
        """
        Validates inputted parameters, implementation pending.
        """
        return None