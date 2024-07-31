# -*- coding: utf-8 -*-

import torch
import pyrenew.math as pmath

def test_asymptotic_properties():
    """
    Check that the calculated
    asymptotic growth rate and
    age distribution given by
    get_asymptotic_growth_rate()
    and get_stable_age_distribution()
    agree with simulated ones from
    just running a process for a
    while.
    """
    R = 1.2
    gi = torch.tensor([0.2, 0.1, 0.2, 0.15, 0.05, 0.025, 0.025, 0.25], dtype=torch.float32)
    A = pmath.get_leslie_matrix(R, gi)

    # check via Leslie matrix multiplication
    x = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    for i in range(1000):
        x_new = torch.matmul(A, x)
        rat_x = torch.sum(x_new) / torch.sum(x)
        x = x_new

    assert torch.allclose(
        rat_x, torch.tensor(pmath.get_asymptotic_growth_rate(R, gi)), atol=1e-5
    ), f'Expected {rat_x} to be close to {pmath.get_asymptotic_growth_rate(R, gi)}'
    
    assert torch.allclose(
        x / torch.sum(x), pmath.get_stable_age_distribution(R, gi), atol=1e-5
    ), f'Expected {x / torch.sum(x)} to be close to {pmath.get_stable_age_distribution(R, gi)}'

    # check via backward-looking convolution
    y = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    for j in range(1000):
        new_pop = torch.dot(y, R * gi)
        rat_y = new_pop / y[0]
        y = torch.cat([new_pop.view(1), y[:-1]])
    
    assert torch.allclose(
        rat_y, torch.tensor(pmath.get_asymptotic_growth_rate(R, gi)), atol=1e-5
    ), f'Expected {rat_y} to be close to {pmath.get_asymptotic_growth_rate(R, gi)}'
    
    assert torch.allclose(
        y / torch.sum(y), pmath.get_stable_age_distribution(R, gi), atol=1e-5
    ), f'Expected {y / torch.sum(y)} to be close to {pmath.get_stable_age_distribution(R, gi)}'

if __name__ == "__main__":
    test_asymptotic_properties()
 