from typing import NamedTuple
import torch
import pyro
import pyro.distributions as dist
from pyrenew.deterministic import DeterministicVariable
from pyrenew.metaclass import RandomVariable

class HospitalAdmissionsSample(NamedTuple):
    """
    A container to hold the output of `latent.HospAdmissions.sample()`.

    Attributes:
    ----------
    infection_hosp_rate : float, optional
        The infection-to-hospitalization rate. Defaults to None.
    latent_hospital_admissions : torch.Tensor or None
        The computed number of hospital admissions. Defaults to None.
    """

    infection_hosp_rate: float | None = None
    latent_hospital_admissions: torch.Tensor | None = None

    def __repr__(self):
        return f"HospitalAdmissionsSample(infection_hosp_rate={self.infection_hosp_rate}, latent_hospital_admissions={self.latent_hospital_admissions})"

class HospitalAdmissions(RandomVariable):
    """
    Latent hospital admissions model using a renewal process for estimating hospital admissions.
    """

    def __init__(
        self,
        infection_to_admission_interval_rv: RandomVariable,
        infect_hosp_rate_rv: RandomVariable,
        latent_hospital_admissions_varname: str = "latent_hospital_admissions",
        day_of_week_effect_rv: RandomVariable | None = None,
        hosp_report_prob_rv: RandomVariable | None = None,
    ):
        """
        Initializes the Hospital Admissions model.
        """

        self.validate(infect_hosp_rate_rv, day_of_week_effect_rv, hosp_report_prob_rv)
        
        self.infection_to_admission_interval_rv = infection_to_admission_interval_rv
        self.infect_hosp_rate_rv = infect_hosp_rate_rv
        self.latent_hospital_admissions_varname = latent_hospital_admissions_varname
        self.day_of_week_effect_rv = day_of_week_effect_rv or DeterministicVariable(1, "weekday_effect")
        self.hosp_report_prob_rv = hosp_report_prob_rv or DeterministicVariable(1, "hosp_report_prob")

    @staticmethod
    def validate(infect_hosp_rate_rv, day_of_week_effect_rv, hosp_report_prob_rv):
        """
        Validates that the provided random variables are of the appropriate type.
        """
        assert isinstance(infect_hosp_rate_rv, RandomVariable), "infect_hosp_rate_rv must be a RandomVariable"
        if day_of_week_effect_rv is not None:
            assert isinstance(day_of_week_effect_rv, RandomVariable), "day_of_week_effect_rv must be a RandomVariable"
        if hosp_report_prob_rv is not None:
            assert isinstance(hosp_report_prob_rv, RandomVariable), "hosp_report_prob_rv must be a RandomVariable"

    def sample(self, latent_infections: torch.Tensor, **kwargs) -> HospitalAdmissionsSample:
        """
        Samples the expected number of hospital admissions based on latent infections.
        """

        infection_hosp_rate = self.infect_hosp_rate_rv.sample()
        infection_hosp_rate_t = infection_hosp_rate * latent_infections

        infection_to_admission_interval = self.infection_to_admission_interval_rv.sample()

        # Using torch's convolve function to compute the convolution
        latent_hospital_admissions = torch.conv1d(
            infection_hosp_rate_t.view(1, 1, -1),
            infection_to_admission_interval.view(1, 1, -1),
            padding='same'
        ).squeeze()

        # Adjustments based on day of the week and hospitalization probability
        day_of_week_effect = self.day_of_week_effect_rv.sample()
        hosp_report_prob = self.hosp_report_prob_rv.sample()

        latent_hospital_admissions *= day_of_week_effect * hosp_report_prob

        # Registering the sample as a deterministic Pyro variable
        pyro.deterministic(self.latent_hospital_admissions_varname, latent_hospital_admissions)

        return HospitalAdmissionsSample(infection_hosp_rate, latent_hospital_admissions)