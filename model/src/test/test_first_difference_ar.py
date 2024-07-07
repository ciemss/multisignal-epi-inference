# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

import torch
import pyro
from pyrenew.process import FirstDifferenceARProcess


# Testing the functionality
def test_fd_ar_can_be_sampled():
    """
    Test the FirstDifferenceARProcess for correct initialization and sampling.
    """
    autoreg = torch.tensor(0.5)
    noise_sd = 0.5
    ar_fd = FirstDifferenceARProcess(autoreg, noise_sd)
    init_val = torch.tensor(50.0)
    init_rate_of_change = torch.tensor(0.25)

    # Set random seed for reproducibility
    pyro.set_rng_seed(62)

    # Sampling
    result = ar_fd.sample(3532, init_val=init_val)
    result_with_init_rate = ar_fd.sample(3532, init_val=init_val, init_rate_of_change=init_rate_of_change)

    assert result.shape == (3532,)
    assert result_with_init_rate.shape == (3532,)

    print("Tests passed!")
