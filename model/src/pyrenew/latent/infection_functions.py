import torch
from pyrenew.convolve import torch_convolve_scanner, torch_double_convolve_scanner, torch_scan
from pyrenew.transformation import ExpTransform
from pyrenew.transformation import IdentityTransform
from torch import Tensor
from typing import Tuple

def compute_infections_from_rt(I0, Rt, reversed_generation_interval_pmf):
    """
    Generate infections according to a renewal process with a time-varying reproduction number R(t)
    """
    # Define the incidence function using the convolution scanner
    incidence_func = torch_convolve_scanner(reversed_generation_interval_pmf, IdentityTransform())

    # Initialize infections with the initial values
    _, all_infections = torch_scan(f=incidence_func, init=I0, xs=Rt)

    return all_infections

def logistic_susceptibility_adjustment(I_raw_t, frac_susceptible, n_population):
    """
    Apply the logistic susceptibility adjustment to a potential new incidence.
    """
    approx_frac_infected = 1 - torch.exp(-I_raw_t / n_population)
    return n_population * frac_susceptible * approx_frac_infected

def compute_infections_from_rt_with_feedback(
    I0: Tensor,
    Rt_raw: Tensor,
    infection_feedback_strength: Tensor,
    reversed_generation_interval_pmf: Tensor,
    reversed_infection_feedback_pmf: Tensor
) -> Tuple[Tensor, Tensor]:
    r"""
    Compute the time series of infections influenced by the reproductive number (Rt),
    with adjustments based on infection feedback, which reflects changes in the
    transmission rate due to past infections.

    This function models the expected number of new infections as a function of the
    effective reproduction number, which itself is adjusted based on recent infection
    history. This approach can capture phenomena such as behavioral changes or
    immunity development in response to the infection spread.

    Parameters
    ----------
    I0 : Tensor
        Initial infections vector, which sets the starting conditions for the
        infection calculation. Each element corresponds to the number of new
        infections at each time step at the start of the simulation.
    Rt_raw : Tensor
        A time series vector of the basic reproduction number (Rt) without feedback.
        Each value represents the expected number of secondary cases produced by
        a single infection in a completely susceptible population.
    infection_feedback_strength : Tensor
        A vector or scalar that scales the influence of past infections on the
        current reproduction number. Positive values increase Rt due to feedback,
        while negative values decrease it, simulating effects like behavioral
        changes or partial immunity.
    reversed_generation_interval_pmf : Tensor
        The probability mass function of the generation interval, reversed so that
        the most recent time points come first. This defines how long it typically
        takes for an infected individual to infect another person.
    reversed_infection_feedback_pmf : Tensor
        A PMF describing how past infections impact the current reproduction number,
        reversed in the same manner as the generation interval PMF. This represents
        the temporal distribution of feedback effects from past infections.

    Returns
    -------
    Tuple[Tensor, Tensor]
        A tuple containing two elements:
        - The first element is a Tensor of the simulated infection counts over time.
        - The second element is a Tensor of the adjusted reproduction numbers over time,
          modified by the infection feedback process.

    Notes
    -----
    The underlying model assumes that the feedback process affects the transmission rate
    through a multiplicative factor that modifies the raw reproduction number based on
    recent infections. The process is formalized by the renewal equation:

    .. math::

        I(t) = Rt(t) * \sum_{\tau=1}^{T_g} I(t-\tau) * g(\tau)

        Rt(t) = Rt_{raw}(t) * exp(\gamma(t) * \sum_{\tau=1}^{T_f} I(t-\tau) * f(\tau))

    where:
    - :math:`I(t)` is the number of new infections at time :math:`t`.
    - :math:`Rt(t)` is the effective reproduction number at time :math:`t`.
    - :math:`\gamma(t)` is the infection feedback strength at time :math:`t`.
    - :math:`g(\tau)` is the probability of transmitting the infection \tau time units after being infected.
    - :math:`f(\tau)` describes the effect of infections \tau time units ago on the current Rt.
    - :math:`T_g` and :math:`T_f` are the lengths of the generation interval and feedback PMFs, respectively.

    This model is particularly useful in epidemiological modeling where feedback mechanisms
    are crucial for understanding and predicting the dynamics of disease spread.
    """
    # Setup the scanner function
    feedback_scanner = torch_double_convolve_scanner(
        arrays_to_convolve=(reversed_infection_feedback_pmf, reversed_generation_interval_pmf),
        transforms=(torch.exp, lambda x: x)  # Assuming torch.exp and identity for transformations
    )

    # Initialize the scan
    init = (I0, (torch.zeros_like(I0), torch.zeros_like(I0)))
    xs = (infection_feedback_strength, Rt_raw)

    # Apply the scan over the inputs
    final_state, outputs = torch_scan(feedback_scanner, init, xs)

    # Extract the adjusted Rt and infections
    infections, Rt_adjustments = outputs
    Rt_adjusted = Rt_adjustments * Rt_raw

    return infections, Rt_adjusted