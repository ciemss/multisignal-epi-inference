# -*- coding: utf-8 -*-
from typing import NamedTuple

import torch
import pyro
import pyro.distributions as dist
from pyrenew.latent.infection_functions import compute_infections_from_rt_with_feedback
from pyrenew.arrayutils import pad_x_to_match_y
from pyrenew.metaclass import RandomVariable


class InfectionsRtFeedbackSample(NamedTuple):
    """
    A container for holding the output from the InfectionsWithFeedback.

    Attributes
    ----------
    post_seed_infections : torch.Tensor | None, optional
        The estimated latent infections. Defaults to None.
    rt : torch.Tensor | None, optional
        The adjusted reproduction number. Defaults to None.
    """

    post_seed_infections: torch.Tensor | None = None
    rt: torch.Tensor | None = None

    def __repr__(self):
        return f"InfectionsSample(post_seed_infections={self.post_seed_infections}, rt={self.rt})"


class InfectionsWithFeedback(RandomVariable):
    r"""Latent infections

    This class computes infections, given Rt, initial infections, and generation
    interval.

    Notes
    -----
    Implements a renewal process that includes feedback effects on the reproduction number R(t).
    """

    def __init__(
        self,
        infection_feedback_strength: RandomVariable,
        infection_feedback_pmf: RandomVariable,
    ) -> None:
        """
        Default constructor for Infections class.

        Parameters
        ----------
        infection_feedback_strength : RandomVariable
            Infection feedback strength.
        infection_feedback_pmf : RandomVariable
            Infection feedback pmf.
        """
        self.infection_feedback_strength = infection_feedback_strength
        self.infection_feedback_pmf = infection_feedback_pmf

    def sample(
        self,
        Rt: torch.Tensor,
        I0: torch.Tensor,
        gen_int: torch.Tensor,
        **kwargs,
    ) -> InfectionsRtFeedbackSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : torch.Tensor
            Reproduction number.
        I0 : torch.Tensor
            Initial infections vector of at least the length of the generation interval PMF.
        gen_int : torch.Tensor
            Generation interval PMF.
        """
        if I0.size(0) < gen_int.size(0):
            raise ValueError(
                "Initial infections vector must be at least as long as the generation interval."
            )

        gen_int_rev = torch.flip(gen_int, [0])
        recent_I0 = I0[-gen_int_rev.size(0):]

        # Sampling inf feedback strength
        inf_feedback_strength = self.infection_feedback_strength.sample()

        # Ensuring inf_feedback_strength spans the Rt length
        if inf_feedback_strength.size(0) == 1:
            inf_feedback_strength = pad_x_to_match_y(
                x=inf_feedback_strength,
                y=Rt,
                fill_value=inf_feedback_strength.item()
            )
        elif inf_feedback_strength.size(0) != Rt.size(0):
            raise ValueError(
                "Infection feedback strength must be of size 1 or the same size as the reproduction number."
            )

        # Sampling inf feedback PMF
        inf_feedback_pmf = self.infection_feedback_pmf.sample()

        inf_fb_pmf_rev = torch.flip(inf_feedback_pmf, [0])

        post_seed_infections, Rt_adj = compute_infections_from_rt_with_feedback(
            I0=recent_I0,
            Rt_raw=Rt,
            infection_feedback_strength=inf_feedback_strength,
            reversed_generation_interval_pmf=gen_int_rev,
            reversed_infection_feedback_pmf=inf_fb_pmf_rev,
        )

        pyro.deterministic("Rt_adjusted", Rt_adj)

        return InfectionsRtFeedbackSample(
            post_seed_infections=post_seed_infections,
            rt=Rt_adj,
        )