"""
Test the InfectionsWithFeedback class
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
import pyrenew.datautils as du
import pyrenew.latent as latent
from jax.typing import ArrayLike
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable


def _infection_w_feedback_alt(
    gen_int: ArrayLike,
    Rt: ArrayLike,
    I0: ArrayLike,
    inf_feedback_strength: ArrayLike,
    inf_feedback_pmf: ArrayLike,
) -> tuple:
    """
    Calculate the infections with feedback.
    Parameters
    ----------
    gen_int : ArrayLike
        Generation interval.
    Rt : ArrayLike
        Reproduction number.
    I0 : ArrayLike
        Initial infections.
    inf_feedback_strength : ArrayLike
        Infection feedback strength.
    inf_feedback_pmf : ArrayLike
        Infection feedback pmf.

    Returns
    -------
    tuple
    """

    Rt, gen_int = du.pad_to_match(Rt, gen_int, fill_value=0.0)
    # gen_int_rev = np.flip(gen_int)

    I0_vec = du.pad_x_to_match_y(I0, Rt, fill_value=0.0)
    I0_vec = np.array(I0_vec)
    inf_feedback_strength = du.pad_x_to_match_y(
        inf_feedback_strength, Rt, fill_value=inf_feedback_strength[0]
    )
    T = len(Rt)

    Rt_adj = np.zeros(T)

    for t in range(T):
        Rt_adj[t] = Rt[t] * np.exp(
            inf_feedback_strength[t]
            * np.dot(I0_vec, np.flip(inf_feedback_pmf))
        )

        I0_vec[t] = Rt_adj[t] * np.dot(I0_vec, np.flip(gen_int))

    return {"infections": I0_vec, "rt": Rt_adj}


def test_infectionsrtfeedback():
    """
    Test the InfectionsWithFeedback matching the Infections class.
    """

    Rt = jnp.array([0.5, 0.6, 0.7, 0.8])
    I0 = jnp.array([1.0])
    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    # By doing the infection feedback strength 0, Rt = Rt_adjusted
    # So infection should be equal in both
    inf_feed_strength = DeterministicVariable(jnp.array([0.0]))
    inf_feedback_pmf = DeterministicPMF(gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with npro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    assert_array_equal(samp1.infections, samp2.infections)
    assert_array_equal(samp1.rt, Rt)

    return None


def test_infectionsrtfeedback_feedback():
    """
    Test the InfectionsWithFeedback with feedback
    """

    Rt = jnp.array([0.5, 0.6, 0.7, 0.8])
    I0 = jnp.array([1.0])
    gen_int = jnp.array([0.25, 0.25, 0.25, 0.25])

    inf_feed_strength = DeterministicVariable(jnp.repeat(0.5, len(Rt)))
    inf_feedback_pmf = DeterministicPMF(gen_int)

    # Test the InfectionsWithFeedback class
    InfectionsWithFeedback = latent.InfectionsWithFeedback(
        infection_feedback_strength=inf_feed_strength,
        infection_feedback_pmf=inf_feedback_pmf,
    )

    infections = latent.Infections()

    with npro.handlers.seed(rng_seed=0):
        samp1 = InfectionsWithFeedback.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

        samp2 = infections.sample(
            gen_int=gen_int,
            Rt=Rt,
            I0=I0,
        )

    res = _infection_w_feedback_alt(
        gen_int=gen_int,
        Rt=Rt,
        I0=I0,
        inf_feedback_strength=inf_feed_strength.sample()[0],
        inf_feedback_pmf=inf_feedback_pmf.sample()[0],
    )

    assert not jnp.array_equal(samp1.infections, samp2.infections)
    assert_array_equal(samp1.infections, res['infections'])
    assert_array_equal(samp1.rt, res['rt'])

    return None


test_infectionsrtfeedback_feedback()
