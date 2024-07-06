# numpydoc ignore=GL08

import torch
from torch import Tensor
from pyrenew.deterministic.deterministic import DeterministicVariable

class DeterministicProcess(DeterministicVariable):
    """
    A deterministic process (degenerate) random variable.
    Useful to pass fixed quantities over time.
    """

    __init__ = DeterministicVariable.__init__

    def sample(
        self,
        duration: int,
        **kwargs,
    ) -> tuple:
        """
        Retrieve the value of the deterministic RV.

        Parameters
        ----------
        duration : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Ignored.

        Returns
        -------
        tuple
            Containing the stored values during construction.
        """
        res, *_ = super().sample(**kwargs)

        dif = duration - res.shape[0]

        if dif > 0:
            return (torch.cat([res, res[-1].repeat(dif)]),)

        return (res[:duration],)
