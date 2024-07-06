# -*- coding: utf-8 -*-
from typing import NamedTuple

import torch
import pyro
from pyro.distributions import Distribution
from pyrenew.arrayutils import pad_x_to_match_y
from pyrenew.metaclass import Model
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel

class HospModelSample(NamedTuple):
    """
    A container for holding the output from `model.HospitalAdmissionsModel.sample()`.

    Attributes
    ----------
    Rt : float | None, optional
        The reproduction number over time. Defaults to None.
    latent_infections : torch.Tensor | None, optional
        The estimated number of new infections over time. Defaults to None.
    infection_hosp_rate : float | None, optional
        The infected hospitalization rate. Defaults to None.
    latent_hosp_admissions : torch.Tensor | None, optional
        The estimated latent hospitalizations. Defaults to None.
    observed_hosp_admissions : torch.Tensor | None, optional
        The sampled or observed hospital admissions. Defaults to None.
    """

    Rt: float | None = None
    latent_infections: torch.Tensor | None = None
    infection_hosp_rate: float | None = None
    latent_hosp_admissions: torch.Tensor | None = None
    observed_hosp_admissions: torch.Tensor | None = None

    def __repr__(self):
        return (
            f"HospModelSample(Rt={self.Rt}, "
            f"latent_infections={self.latent_infections}, "
            f"infection_hosp_rate={self.infection_hosp_rate}, "
            f"latent_hosp_admissions={self.latent_hosp_admissions}, "
            f"observed_hosp_admissions={self.observed_hosp_admissions}"
        )

class HospitalAdmissionsModel(Model):
    """
    Hospital Admissions Model (BasicRenewal + HospitalAdmissions)
    """

    def __init__(
        self,
        latent_hosp_admissions_rv: Distribution,
        latent_infections_rv: Distribution,
        gen_int_rv: Distribution,
        I0_rv: Distribution,
        Rt_process_rv: Distribution,
        hosp_admission_obs_process_rv: Distribution,
    ) -> None:
        """
        Initialize the Hospital Admissions Model
        """
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int_rv=gen_int_rv,
            I0_rv=I0_rv,
            latent_infections_rv=latent_infections_rv,
            Rt_process_rv=Rt_process_rv,
        )

        self.latent_hosp_admissions_rv = latent_hosp_admissions_rv
        self.hosp_admission_obs_process_rv = hosp_admission_obs_process_rv

    def sample(
        self,
        n_timepoints_to_simulate: int = None,
        data_observed_hosp_admissions: torch.Tensor = None,
        **kwargs,
    ) -> HospModelSample:
        """
        Sample from the Hospital Admissions model
        """
        if n_timepoints_to_simulate is None and data_observed_hosp_admissions is None:
            raise ValueError("Must specify either n_timepoints_to_simulate or provide data_observed_hosp_admissions.")

        n_timepoints = n_timepoints_to_simulate if n_timepoints_to_simulate is not None else len(data_observed_hosp_admissions)

        # Sample initial quantities from the basic renewal model
        basic_model_sample = self.basic_renewal.sample(n_timepoints=n_timepoints, **kwargs)

        # Sample the latent hospital admissions
        latent_hosp_admissions = pyro.sample("latent_hospital_admissions", self.latent_hosp_admissions_rv)

        observed_hosp_admissions = None
        if self.hosp_admission_obs_process_rv is not None:
            observed_hosp_admissions = pyro.sample("observed_hospital_admissions", self.hosp_admission_obs_process_rv, obs=data_observed_hosp_admissions)

        return HospModelSample(
            Rt=basic_model_sample.Rt,
            latent_infections=basic_model_sample.latent_infections,
            latent_hosp_admissions=latent_hosp_admissions,
            observed_hosp_admissions=observed_hosp_admissions
        )