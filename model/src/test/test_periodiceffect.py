"""
Test the periodiceffect module
"""

import jax.numpy as jnp
import numpy as np
import numpyro as npro
from numpy.testing import assert_array_equal
from pyrenew.deterministic import DeterministicVariable
from pyrenew.process import DayOfWeekEffect, PeriodicEffect


def test_periodiceffect() -> None:
    """Checks basic functionality of the process"""

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rv = DeterministicVariable(x, name="weekly-sample")

    params = {
        "offset": 0,
        "quantity_to_broadcast": rv,
        "period_size": 7,
        "t_start": 0,
        "t_unit": 1,
    }

    duration = 30

    pe = PeriodicEffect(**params)

    np.random.seed(223)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        ans = pe.sample(duration=duration).value

    # Checking that the shape of the sampled Rt is correct
    assert ans.shape == (duration,)

    # Checking that the sampled Rt is constant every 7 days
    for i in range(0, 28, 7):
        assert_array_equal(ans[i : i + 7], x)

    # Checking start off a different day of the week
    np.random.seed(223)
    params["offset"] = 5
    pe = PeriodicEffect(**params)
    with npro.handlers.seed(rng_seed=np.random.randint(1, 600)):
        ans2 = pe.sample(duration=duration).value

    # Checking that the shape of the sampled Rt is correct
    assert ans2.shape == (duration,)

    # This time series should be the same as the previous one, but shifted by
    # 5 days
    assert_array_equal(ans[5:], ans2[:-5])

    return None


def test_weeklyeffect() -> None:
    """Checks basic functionality of the process"""

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    rv = DeterministicVariable(x, name="weekly-sample")

    params = {
        "offset": 2,
        "quantity_to_broadcast": rv,
        "period_size": 7,
        "t_start": 0,
        "t_unit": 1,
    }

    params2 = {
        "offset": 2,
        "quantity_to_broadcast": rv,
        "t_start": 0,
    }

    duration = 30

    pe = PeriodicEffect(**params)
    pe2 = DayOfWeekEffect(**params2)

    ans1 = pe.sample(duration=duration).value
    ans2 = pe2.sample(duration=duration).value

    assert_array_equal(ans1, ans2)

    return None
