import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from typing import NamedTuple
from pyrenew.convolve import torch_convolve


class HospitalAdmissionsSample(NamedTuple):
    infection_hosp_rate: float | None = None
    latent_hospital_admissions: torch.Tensor | None = None

    def __repr__(self):
        return f"HospitalAdmissionsSample(infection_hosp_rate={self.infection_hosp_rate}, latent_hospital_admissions={self.latent_hospital_admissions})"

class HospitalAdmissions(PyroModule):
    def __init__(self, infection_to_admission_interval_rv, infect_hosp_rate_rv, latent_hospital_admissions_varname="latent_hospital_admissions", day_of_week_effect_rv=None, hosp_report_prob_rv=None):
        super().__init__()
        if day_of_week_effect_rv is None:
            day_of_week_effect_rv = torch.tensor(1.0)  # Default value if not provided
        if hosp_report_prob_rv is None:
            hosp_report_prob_rv = torch.tensor(1.0)  # Assume full reporting if not provided

        self.latent_hospital_admissions_varname = latent_hospital_admissions_varname
        self.infect_hosp_rate_rv = PyroSample(dist.Delta(infect_hosp_rate_rv))
        self.day_of_week_effect_rv = PyroSample(dist.Delta(day_of_week_effect_rv))
        self.hosp_report_prob_rv = PyroSample(dist.Delta(hosp_report_prob_rv))
        self.infection_to_admission_interval_rv = infection_to_admission_interval_rv

    def sample(self, latent_infections: Tensor, **kwargs) -> HospitalAdmissionsSample:
        """
        Samples from the observation process

        Parameters
        ----------
        latent_infections : Tensor
            Latent infections.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, should there be any.

        Returns
        -------
        HospitalAdmissionsSample
        """
        infection_hosp_rate, *_ = self.infect_hosp_rate_rv.sample(**kwargs)
        infection_hosp_rate_t = infection_hosp_rate * latent_infections

        # Sample the infection to admission interval
        infection_to_admission_interval, *_ = self.infection_to_admission_interval_rv.sample(**kwargs)

        # Using torch_convolve to compute the convolution
        latent_hospital_admissions = torch_convolve(
            infection_hosp_rate_t,
            infection_to_admission_interval,
            mode='full'
        )

        # Slice the result to match the size of the original infections tensor
        latent_hospital_admissions = latent_hospital_admissions[:len(infection_hosp_rate_t)]

        # Applying the day of the week effect
        day_of_week_effect = self.day_of_week_effect_rv.sample(**kwargs)
        latent_hospital_admissions *= day_of_week_effect

        # Applying probability of hospitalization effect
        hosp_report_prob = self.hosp_report_prob_rv.sample(**kwargs)
        latent_hospital_admissions *= hosp_report_prob

        # Registering the computed hospital admissions as a deterministic variable
        pyro.deterministic(self.latent_hospital_admissions_varname, latent_hospital_admissions)

        return HospitalAdmissionsSample(infection_hosp_rate, latent_hospital_admissions)

    def __repr__(self):
        return f"HospitalAdmissions({self.latent_hospital_admissions_varname})"