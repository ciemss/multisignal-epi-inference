# -*- coding: utf-8 -*-
import pyro
from typing import NamedTuple
import torch
from torch import Tensor as ArrayLike
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.model.rtinfectionsrenewalmodel import RtInfectionsRenewalModel

class HospModelSample(NamedTuple):
    """
    A container for holding the output from `model.HospitalAdmissionsModel.sample()`.
    """
    Rt: float | None = None
    latent_infections: ArrayLike | None = None
    infection_hosp_rate: float | None = None
    latent_hosp_admissions: ArrayLike | None = None
    observed_hosp_admissions: ArrayLike | None = None

class HospitalAdmissionsModel(Model):
    """
    Hospital Admissions Model that combines renewal infection modeling with hospital admission modeling.
    """
    def __init__(self, latent_hosp_admissions_rv: RandomVariable, latent_infections_rv: RandomVariable,
                 gen_int_rv: RandomVariable, I0_rv: RandomVariable, Rt_process_rv: RandomVariable,
                 hosp_admission_obs_process_rv: RandomVariable) -> None:
        self.basic_renewal = RtInfectionsRenewalModel(
            gen_int_rv=gen_int_rv, I0_rv=I0_rv, latent_infections_rv=latent_infections_rv,
            infection_obs_process_rv=None, Rt_process_rv=Rt_process_rv
        )
        HospitalAdmissionsModel.validate(latent_hosp_admissions_rv, hosp_admission_obs_process_rv)
        self.latent_hosp_admissions_rv = latent_hosp_admissions_rv
        self.hosp_admission_obs_process_rv = hosp_admission_obs_process_rv

    @staticmethod
    def validate(latent_hosp_admissions_rv, hosp_admission_obs_process_rv) -> None:
        """
        Validates that the inputs are instances of RandomVariable.
        """
        assert isinstance(latent_hosp_admissions_rv, RandomVariable), "Invalid type for latent_hosp_admissions_rv"
        assert isinstance(hosp_admission_obs_process_rv, RandomVariable), "Invalid type for hosp_admission_obs_process_rv"

    def sample(self, n_timepoints_to_simulate: int | None = None,
               data_observed_hosp_admissions: ArrayLike | None = None, padding: int = 0, **kwargs) -> HospModelSample:
        """
        Samples from the hospital admissions model, using both renewal and observation processes.
        """
        if n_timepoints_to_simulate is None and data_observed_hosp_admissions is None:
            raise ValueError("Specify either n_timepoints_to_simulate or data_observed_hosp_admissions.")
        elif n_timepoints_to_simulate is not None and data_observed_hosp_admissions is not None:
            raise ValueError("Cannot specify both n_timepoints_to_simulate and data_observed_hosp_admissions.")
        n_timepoints = n_timepoints_to_simulate if n_timepoints_to_simulate is not None else len(data_observed_hosp_admissions)
        
        basic_model = self.basic_renewal.sample(n_timepoints_to_simulate=n_timepoints, padding=padding, **kwargs)
        latent_hosp_admissions_sample = self.latent_hosp_admissions_rv.sample(latent_infections=basic_model.latent_infections, **kwargs)
        observed_hosp_admissions = None
        if self.hosp_admission_obs_process_rv is not None and data_observed_hosp_admissions is not None:
            observed_hosp_admissions = self.hosp_admission_obs_process_rv.sample(mu=latent_hosp_admissions_sample.latent_hosp_admissions, obs=data_observed_hosp_admissions, **kwargs)
        
        return HospModelSample(
            Rt=basic_model.Rt,
            latent_infections=basic_model.latent_infections,
            infection_hosp_rate=latent_hosp_admissions_sample.infection_hosp_rate,
            latent_hosp_admissions=latent_hosp_admissions_sample.latent_hosp_admissions,
            observed_hosp_admissions=observed_hosp_admissions
        )