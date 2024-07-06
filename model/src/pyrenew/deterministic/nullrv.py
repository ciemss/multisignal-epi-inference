# numpydoc ignore=GL08

from typing import Union, List, Tuple
import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray, List[float], Tuple[float, ...], float, int]

class NullVariable:
    """A null (degenerate) random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """
        self.validate()

    @staticmethod
    def validate() -> None:
        """
        Not used

        Returns
        -------
        None
        """
        return None

    def sample(
        self,
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the Null (None)

        Parameters
        ----------
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing None.
        """

        return (None,)


class NullProcess(NullVariable):
    """A null random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """
        super().__init__()

    def sample(
        self,
        duration: int,
        **kwargs,
    ) -> tuple:
        """Retrieve the value of the Null (None)

        Parameters
        ----------
        duration : int
            Number of timepoints to sample (ignored).
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing None.
        """

        return (None,)


class NullObservation(NullVariable):
    """A null observation random variable. Sampling returns None."""

    def __init__(self) -> None:
        """Default constructor

        Returns
        -------
        None
        """
        super().__init__()

    def sample(
        self,
        mu: TensorLike,
        obs: Union[TensorLike, None] = None,
        name: Union[str, None] = None,
        **kwargs,
    ) -> tuple:
        """
        Retrieve the value of the Null (None)

        Parameters
        ----------
        mu : TensorLike
            Unused parameter, represents mean of non-null distributions
        obs : TensorLike, optional
            Observed data. Defaults to None.
        name : str, optional
            Name of the random variable. Defaults to None.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample calls, should there be any.

        Returns
        -------
        tuple
            Containing None.
        """

        return (None,)
