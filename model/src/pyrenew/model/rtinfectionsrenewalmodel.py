# -*- coding: utf-8 -*-
import torch
import pyro
import pyro.distributions as dist
from typing import NamedTuple, Optional

# Define the output class of the RtInfectionsRenewalModel
class RtInfectionsRenewalSample(NamedTuple):
    """
    A container for holding the output from `model.RtInfectionsRenewalModel.sample()`.

    Attributes
    ----------
    Rt : torch.Tensor | None, optional
        The reproduction number over time.
    latent_infections : torch.Tensor | None, optional
        The estimated latent infections.
    observed_infections : torch.Tensor | None, optional
        The sampled infections.
    """

    Rt: Optional[torch.Tensor] = None
    latent_infections: Optional[torch.Tensor] = None
    observed_infections: Optional[torch.Tensor] = None

    def __repr__(self):
        return (
            f"RtInfectionsRenewalSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"observed_infections={self.observed_infections})"
        )

class RtInfectionsRenewalModel:
    """
    Basic Renewal Model (Infections + Rt)
    """

    def __init__(self, latent_infections_rv, gen_int_rv, I0_rv, Rt_process_rv, infection_obs_process_rv=None):
        """
        Initialize the Basic Renewal Model components.
        """
        self.gen_int_rv = gen_int_rv
        self.I0_rv = I0_rv
        self.latent_infections_rv = latent_infections_rv
        self.Rt_process_rv = Rt_process_rv
        self.infection_obs_process_rv = infection_obs_process_rv or dist.Delta(0)

    def sample(self, n_timepoints: Optional[int] = None, observed_data: Optional[torch.Tensor] = None):
        """
        Sample from the Basic Renewal Model.

        Parameters:
        -----------
        n_timepoints : int, optional
            Number of timepoints to sample.
        observed_data : torch.Tensor, optional
            Observed infections data.

        Returns:
        --------
        RtInfectionsRenewalSample
            Sampled model outputs.
        """
        if observed_data is not None:
            n_timepoints = observed_data.size(0)

        # Sample from the Rt process
        Rt = pyro.sample("Rt", self.Rt_process_rv.expand([n_timepoints]))

        # Get the generation interval
        gen_int = pyro.sample("gen_int", self.gen_int_rv)

        # Sample initial infections
        I0 = pyro.sample("I0", self.I0_rv)

        # Sample from the latent infection process
        latent_infections = pyro.sample("latent_infections", self.latent_infections_rv(Rt=Rt, gen_int=gen_int, I0=I0))

        # Optionally sample observed infections
        observed_infections = pyro.sample("observed_infections", self.infection_obs_process_rv, obs=observed_data)

        return RtInfectionsRenewalSample(Rt=Rt, latent_infections=latent_infections, observed_infections=observed_infections)