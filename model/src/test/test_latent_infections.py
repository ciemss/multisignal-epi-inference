# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import torch
import numpy as np
import pytest
from pyro.infer import Predictive
import pyro
import pyro.distributions as dist
from pyrenew.latent import Infections
from pyrenew.process import RtRandomWalkProcess
import pyrenew.transformation as t

def test_infections_as_deterministic():
    """
    Test that the Infections class samples the same infections when
    the same seed is used.
    """
    torch.manual_seed(223)
    rt = RtRandomWalkProcess(
        Rt0_dist=dist.HalfNormal(scale=0.2),  # Assuming use of HalfNormal for simplicity as a substitute for TruncatedNormal
        Rt_transform=t.ExpTransform(),  # Custom transform to handle PyTorch
        Rt_rw_dist=dist.Normal(0, 0.025),
    )

    # Pyro way to handle seeds
    pyro.set_rng_seed(np.random.randint(1, 600))
    sim_rt, *_ = rt.sample(n_timepoints=30)

    gen_int = torch.tensor([0.25, 0.25, 0.25, 0.25])

    inf1 = Infections()

    obs = dict(
        Rt=sim_rt,
        I0=torch.zeros(gen_int.size()),
        gen_int=gen_int,
    )

    pyro.set_rng_seed(np.random.randint(1, 600))
    inf_sampled1 = inf1.sample(**obs)
    inf_sampled2 = inf1.sample(**obs)

    np.testing.assert_array_equal(
        inf_sampled1.post_seed_infections.numpy(), inf_sampled2.post_seed_infections.numpy()
    )

    # Check that Initial infections vector must be at least as long as the generation interval.
    with pytest.raises(ValueError):
        obs["I0"] = torch.tensor([1.0])
        inf1.sample(**obs)