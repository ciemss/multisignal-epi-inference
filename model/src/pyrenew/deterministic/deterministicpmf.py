# -*- coding: utf-8 -*-
# numpydoc ignore=GL08

from jax.typing import ArrayLike
from pyrenew.deterministic.deterministic import DeterministicVariable
from pyrenew.distutil import validate_discrete_dist_vector
from pyrenew.metaclass import RandomVariable


class DeterministicPMF(RandomVariable):
    """
    A deterministic (degenerate) random variable that represents a PMF."""

    def __init__(
        self,
        vars: ArrayLike,
        label: str = "a_random_variable",
        tol: float = 1e-10,
    ) -> None:
        """
        Default constructor

        Automatically checks that the elements in `vars` can be indeed
        considered to be a PMF by calling
        pyrenew.distutil.validate_discrete_dist_vector on each one of its
        entries.

        Parameters
        ----------
        vars : tuple
            A tuple with arraylike objects.
        label : str, optional
            A label to assign to the process. Defaults to "a_random_variable"
        tol : float, optional
            Passed to pyrenew.distutil.validate_discrete_dist_vector. Defaults
            to 1e-20.

        Returns
        -------
        None
        """
        vars = validate_discrete_dist_vector(
            discrete_dist=vars,
            tol=tol,
        )

        self.basevar = DeterministicVariable(vars, label)

        return None

    @staticmethod
    def validate(vars: ArrayLike) -> None:
        """
        Validates inputted to DeterministicPMF

        Parameters
        ----------
        vars : ArrayLike
            An ArrayLike object.

        Returns
        -------
        None
        """
        return None

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """
        Retrieves the deterministic PMF

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal `sample()`
            calls, if any

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """

        return self.basevar.sample(**kwargs)
