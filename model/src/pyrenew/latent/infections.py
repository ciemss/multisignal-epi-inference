# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import NamedTuple

import torch
import pyrenew.latent.infection_functions as inf
from pyrenew.metaclass import RandomVariable


class InfectionsSample(NamedTuple):
    """
    A container for holding the output from `latent.Infections.sample()`.

    Attributes
    ----------
    post_seed_infections : torch.Tensor | None, optional
        The estimated latent infections. Defaults to None.
    """

    post_seed_infections: torch.Tensor | None = None

    def __repr__(self):
        return f"InfectionsSample(post_seed_infections={self.post_seed_infections})"


class Infections(RandomVariable):
    r"""Latent infections

    This class samples infections given Rt,
    initial infections, and generation interval.

    Notes
    -----
    The mathematical model is given by:

    .. math::

            I(t) = R(t) \times \sum_{\tau < t} I(\tau) g(t-\tau)

    where :math:`I(t)` is the number of infections at time :math:`t`,
    :math:`R(t)` is the reproduction number at time :math:`t`, and
    :math:`g(t-\tau)` is the generation interval.
    """

    @staticmethod
    def validate() -> None:
        return None

    def sample(
        self,
        Rt: torch.Tensor,
        I0: torch.Tensor,
        gen_int: torch.Tensor,
        **kwargs,
    ) -> InfectionsSample:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : torch.Tensor
            Reproduction number.
        I0 : torch.Tensor
            Initial infections vector
            of the same length as the
            generation interval.
        gen_int : torch.Tensor
            Generation interval pmf vector.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """
        if I0.size(0) < gen_int.size(0):
            raise ValueError(
                "Initial infections vector must be at least as long as "
                "the generation interval. "
                f"Initial infections vector length: {I0.size(0)}, "
                f"generation interval length: {gen_int.size(0)}."
            )

        gen_int_rev = torch.flip(gen_int, [0])
        recent_I0 = I0[-gen_int_rev.size(0):]

        post_seed_infections = inf.compute_infections_from_rt(
            I0=recent_I0,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        return InfectionsSample(post_seed_infections)