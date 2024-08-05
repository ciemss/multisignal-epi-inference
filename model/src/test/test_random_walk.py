# numpydoc ignore=GL08

import torch
import pyro
import pyro.distributions as dist
from numpy.testing import assert_almost_equal
from pyrenew.process import SimpleRandomWalkProcess


def test_rw_can_be_sampled():
    """
    Check that a simple random walk
    can be initialized and sampled from
    """
    rw_normal = SimpleRandomWalkProcess(dist.Normal(0., 1.))

    pyro.set_rng_seed(rng_seed=62)
    # can sample with and without inits
    ans0 = rw_normal.sample(3532, init=torch.tensor([50.0]))
    ans1 = rw_normal.sample(5023)

    # check that the samples are of the right shape
    assert ans0[0].shape == (3532,)
    assert ans1[0].shape == (5023,)


def test_rw_samples_correctly_distributed():
    """
    Check that a simple random walk has steps
    distributed according to the target distribution
    """

    n_samples = 10000
    for step_mean, step_sd in zip(
        [0., 2.253, -3.2521, 1052., 1e-6], [1., 0.025, 3., 1., 0.02]
    ):
        step_sd = torch.tensor(step_sd)
        rw_normal = SimpleRandomWalkProcess(dist.Normal(step_mean, step_sd))
        init_arr = torch.tensor([532.0], dtype=torch.float)
        pyro.set_rng_seed(rng_seed=62)
        samples, *_ = rw_normal.sample(n_samples, init=init_arr)

        # Checking the shape
        assert samples.shape == (n_samples,)

        # diffs should not be greater than
        # 5 sigma
        diffs = samples[1:] - samples[:-1]
        assert torch.all(torch.abs(diffs - step_mean) < 5 * step_sd)

        # sample mean of diffs should be
        # approximately equal to the
        # step mean, according to
        # the Law of Large Numbers
        deviation_threshold = 4 * torch.sqrt((step_sd**2) / n_samples)
        assert torch.abs(torch.mean(diffs) - step_mean) < deviation_threshold

        # sample sd of diffs
        # should be approximately equal
        # to the step sd
        assert torch.abs(torch.log(torch.std(diffs) / step_sd)) < torch.log(torch.tensor([1.1]))

        # first value should be the init value
        assert_almost_equal(samples[0], init_arr)
