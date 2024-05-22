# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from typing import NamedTuple

import jax.numpy as jnp
import numpyro as npro
import pyrenew.latent.infection_functions as inf
from numpy.typing import ArrayLike
from pyrenew.metaclass import RandomVariable


class InfectionsSample(NamedTuple):
    """
    A container for holding the output from the InfectionsSample.

    Attributes
    ----------
    infections : ArrayLike | None, optional
        The estimated latent infections. Defaults to None.
    """

    infections: ArrayLike | None = None

    def __repr__(self):
        return f"InfectionsSample(infections={self.infections})"


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

    def __init__(
        self,
        infections_mean_varname: str = "latent_infections",
    ) -> None:
        """
        Default constructor for Infections class.

        Parameters
        ----------
        infections_mean_varname : str, optional
            Name to be assigned to the deterministic variable in the model.
            Defaults to "latent_infections".

        Returns
        -------
        None
        """

        self.infections_mean_varname = infections_mean_varname

        return None

    @staticmethod
    def validate() -> None:  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        Rt: ArrayLike,
        I0: ArrayLike,
        gen_int: ArrayLike,
        **kwargs,
    ) -> tuple:
        """
        Samples infections given Rt, initial infections, and generation
        interval.

        Parameters
        ----------
        Rt : ArrayLike
            Reproduction number.
        I0 : ArrayLike
            Initial infections.
        gen_int : ArrayLike
            Generation interval.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal
            sample calls, should there be any.

        Returns
        -------
        InfectionsSample
            Named tuple with "infections".
        """

        gen_int_rev = jnp.flip(gen_int)

        n_lead = gen_int_rev.size - 1
        I0_vec = jnp.hstack([jnp.zeros(n_lead), I0])

        all_infections = inf.sample_infections_rt(
            I0=I0_vec,
            Rt=Rt,
            reversed_generation_interval_pmf=gen_int_rev,
        )

        npro.deterministic(self.infections_mean_varname, all_infections)

        return InfectionsSample(all_infections)
