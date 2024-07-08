import torch
from pyrenew.latent import infection_functions as inf
from torch.testing import assert_allclose

def test_compute_infections_from_rt_with_feedback():
    """
    Test that the implementation of infection feedback is as expected.

    If feedback is zero, results should be equivalent to compute_infections_from_rt
    and Rt_adjusted should be Rt_raw.
    """

    gen_ints = [
        torch.tensor([0.25, 0.5, 0.25]),
        torch.tensor([1.0]),
        torch.ones(35) / 35,
    ]

    inf_pmfs = [torch.ones_like(x) for x in gen_ints]

    I0s = [
        torch.tensor([0.235, 6.523, 100052.0]),
        torch.tensor([5.0]),
        3.5235 * torch.ones(35),
    ]

    Rts_raw = [
        torch.tensor([1.25, 0.52, 23.0, 1.0]),
        torch.ones(500),
        torch.zeros(253),
    ]

    for I0, gen_int, inf_pmf in zip(I0s, gen_ints, inf_pmfs):
        for Rt_raw in Rts_raw:
            infs_feedback, Rt_adj = inf.compute_infections_from_rt_with_feedback(
                I0, Rt_raw, torch.zeros_like(Rt_raw), gen_int, inf_pmf
            )

            assert_allclose(
                inf.compute_infections_from_rt(I0, Rt_raw, gen_int),
                infs_feedback
            )

            assert_allclose(Rt_adj, Rt_raw)

