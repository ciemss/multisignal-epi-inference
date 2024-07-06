# -*- coding: utf-8 -*-
import pyro
import pyro.distributions as dist
from pyrenew.latent.infection_seeding_method import InfectionSeedMethod
from pyrenew.metaclass import RandomVariable


class InfectionSeedingProcess(RandomVariable):
    """Generate an initial infection history."""

    def __init__(
        self,
        name: str,
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod,
        t_unit: int,
        t_start: int | None = None
    ) -> None:
        """Default class constructor for InfectionSeedingProcess.

        Parameters
        ----------
        name : str
            A name to assign to the RandomVariable.
        I_pre_seed_rv : RandomVariable
            A RandomVariable representing the number of infections that occur at some time before the renewal process begins.
        infection_seed_method : InfectionSeedMethod
            An `InfectionSeedMethod` that generates the seed infections for the renewal process.
        t_unit : int
            The unit of time for the time series passed to `RandomVariable.set_timeseries`.
        t_start : int | None, optional
            The relative starting time of the time series. If `None`, the relative starting time is set to `-infection_seed_method.n_timepoints`.

        Notes
        -----
        The relative starting time of the time series (`t_start`) is set to `-infection_seed_method.n_timepoints`.
        """
        InfectionSeedingProcess.validate(I_pre_seed_rv, infection_seed_method)

        self.I_pre_seed_rv = I_pre_seed_rv
        self.infection_seed_method = infection_seed_method
        self.name = name
        self.t_start = t_start if t_start is not None else -infection_seed_method.n_timepoints
        self.t_unit = t_unit

    @staticmethod
    def validate(
        I_pre_seed_rv: RandomVariable,
        infection_seed_method: InfectionSeedMethod
    ) -> None:
        """Validate the input arguments to the InfectionSeedingProcess class constructor.

        Parameters
        ----------
        I_pre_seed_rv : RandomVariable
            A random variable representing the number of infections that occur at some time before the renewal process begins.
        infection_seed_method : InfectionSeedMethod
            A method to generate the seed infections.
        """
        if not isinstance(I_pre_seed_rv, RandomVariable):
            raise TypeError(f"I_pre_seed_rv must be an instance of RandomVariable, got {type(I_pre_seed_rv)}")
        if not isinstance(infection_seed_method, InfectionSeedMethod):
            raise TypeError(f"infection_seed_method must be an instance of InfectionSeedMethod, got {type(infection_seed_method)}")

    def sample(self) -> tuple:
        """Sample the infection seeding process.

        Returns
        -------
        tuple
            A tuple where the only element is an array with the number of seeded infections at each time point.
        """
        I_pre_seed = self.I_pre_seed_rv.sample()
        infection_seeding = self.infection_seed_method(I_pre_seed)
        pyro.deterministic(self.name, infection_seeding)

        return (infection_seeding,)
    