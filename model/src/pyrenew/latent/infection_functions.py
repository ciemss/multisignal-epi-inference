import torch
from pyrenew.convolve import torch_convolve_scanner, torch_double_convolve_scanner, torch_scan
from pyrenew.transformation import ExpTransform
from pyrenew.transformation import IdentityTransform

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

def compute_infections_from_rt_with_feedback(I0, Rt_raw, infection_feedback_strength, reversed_generation_interval_pmf, reversed_infection_feedback_pmf):
    """
    Generate infections according to a renewal process with infection feedback
    """
    # Define the feedback scanner using the convolution scanner
    feedback_scanner =torch_double_convolve_scanner(
        arrays_to_convolve=(reversed_infection_feedback_pmf, reversed_generation_interval_pmf),
        transforms=(ExpTransform(), IdentityTransform())
    )

    # Initialize infections with the initial values
    infections = I0.clone()
    Rt_adjusted = Rt_raw.clone()

    for i in range(len(Rt_raw)):
        feedback_strength = infection_feedback_strength if torch.is_tensor(infection_feedback_strength) else infection_feedback_strength[i]
        infections, R_adjustment = feedback_scanner(infections, (feedback_strength, Rt_raw[i]))
        Rt_adjusted[i] = R_adjustment * Rt_raw[i]

    return infections, Rt_adjusted