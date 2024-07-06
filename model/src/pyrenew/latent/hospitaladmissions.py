import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroSample

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

    def sample(self, latent_infections, **kwargs):
        infection_hosp_rate = self.infect_hosp_rate_rv.sample()
        infection_hosp_rate_t = infection_hosp_rate * latent_infections

        infection_to_admission_interval = self.infection_to_admission_interval_rv.sample()

        latent_hospital_admissions = torch.convolve(infection_hosp_rate_t, infection_to_admission_interval, mode='full')[:len(infection_hosp_rate_t)]

        # Apply day of the week and reporting probability effects
        latent_hospital_admissions *= self.day_of_week_effect_rv.sample()
        latent_hospital_admissions *= self.hosp_report_prob_rv.sample()

        pyro.deterministic(self.latent_hospital_admissions_varname, latent_hospital_admissions)

        return HospitalAdmissionsSample(infection_hosp_rate, latent_hospital_admissions)

    def __repr__(self):
        return f"HospitalAdmissions({self.latent_hospital_admissions_varname})"