import torch
from torch import tensor

def validate_discrete_dist_vector(discrete_dist):
    """
    Validate that a vector represents a discrete
    probability distribution. Must sum to 1 and have non-negative entries.
    """
    discrete_dist = discrete_dist.flatten()
    if not torch.all(discrete_dist >= 0):
        raise ValueError("All elements must be non-negative.")
    if not torch.isclose(torch.sum(discrete_dist), tensor(1.0)):
        raise ValueError("The elements must sum to 1.")

def get_leslie_matrix(R, generation_interval_pmf):
    """
    Create the Leslie matrix corresponding to a basic renewal process.
    """
    validate_discrete_dist_vector(generation_interval_pmf)
    gen_int_len = generation_interval_pmf.numel()
    aging_matrix = torch.cat([
        torch.eye(gen_int_len - 1),
        torch.zeros(gen_int_len - 1).unsqueeze(1)
    ], dim=1)

    return torch.cat([R * generation_interval_pmf.unsqueeze(0), aging_matrix], dim=0)

def get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf):
    """
    Get the asymptotic growth rate and stable age distribution of the renewal process.
    """
    L = get_leslie_matrix(R, generation_interval_pmf)
    eigenvals, eigenvecs = torch.linalg.eig(L)
    d = torch.argmax(torch.abs(eigenvals))  # index of dominant eigenvalue
    d_vec, d_val = eigenvecs[:, d], eigenvals[d]
    d_vec_real = torch.real(d_vec)
    d_val_real = torch.real(d_val)
    if torch.any(d_vec_real.imag != 0) or torch.any(d_val_real.imag != 0):
        raise ValueError("Complex parts detected in outputs.")
    d_vec_norm = d_vec_real / torch.sum(d_vec_real)
    return d_val_real.item(), d_vec_norm

def get_stable_age_distribution(R, generation_interval_pmf):
    """
    Get the stable age distribution for a renewal process.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[1]

def get_asymptotic_growth_rate(R, generation_interval_pmf):
    """
    Get the asymptotic growth rate for a renewal process.
    """
    return get_asymptotic_growth_rate_and_age_dist(R, generation_interval_pmf)[0]