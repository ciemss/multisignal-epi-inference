import torch
import pyro
import pyro.distributions as dist
from pyrenew.transformation import Transform
from pyrenew.metaclass import RandomVariable
from pyrenew.process.simplerandomwalk import SimpleRandomWalkProcess


class RtRandomWalkProcess(RandomVariable):
    r"""Rt Randomwalk Process

    Notes
    -----
    The process is defined as follows:

    .. math::
        Rt(0) &\sim \text{Rt0_dist} \\
        Rt(t) &\sim \text{Rt_transform}(\text{Rt_transformed_rw}(t))
    """

    def __init__(
        self,
        Rt0_dist: dist.Distribution,
        Rt_rw_dist: dist.Distribution,
        Rt_transform: Transform | None = None,
    ) -> None:
        """
        Default constructor

        Parameters
        ----------
        Rt0_dist : dist.Distribution
            Initial distribution of Rt.
        Rt_rw_dist : dist.Distribution
            Random walk process.
        Rt_transform : Transform, optional
            Transformation applied to the sampled Rt0. If None, the identity
            transformation is used.
        """
        if Rt_transform is None:
            Rt_transform = Transform()  # Assuming Transform() is analogous to IdentityTransform

        self.Rt0_dist = Rt0_dist
        self.Rt_transform = Rt_transform
        self.Rt_rw_dist = Rt_rw_dist

    @staticmethod
    def validate(
        Rt0_dist: dist.Distribution,
        Rt_transform: Transform,
        Rt_rw_dist: dist.Distribution,
    ) -> None:
        """
        Validates Rt0_dist, Rt_transform, and Rt_rw_dist.

        Parameters
        ----------
        Rt0_dist : dist.Distribution
            Initial distribution of Rt, expected dist.Distribution
        Rt_transform : Transform
            Transformation applied to the sampled Rt0.
        Rt_rw_dist : dist.Distribution
            Random walk process, expected dist.Distribution.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If Rt0_dist or Rt_rw_dist are not instances of dist.Distribution or if
            Rt_transform is not an instance of Transform.
        """
        if not isinstance(Rt0_dist, dist.Distribution):
            raise TypeError("Rt0_dist must be an instance of pyro.distributions.Distribution")
        if not isinstance(Rt_transform, Transform):
            raise TypeError("Rt_transform must be an instance of pyrenew.transformation.Transform")
        if not isinstance(Rt_rw_dist, dist.Distribution):
            raise TypeError("Rt_rw_dist must be an instance of pyro.distributions.Distribution")
            
    def sample(
        self,
        n_timepoints: int,
        **kwargs,
    ) -> tuple:
        """
        Generate samples from the process

        Parameters
        ----------
        n_timepoints : int
            Number of timepoints to sample.
        **kwargs : dict, optional
            Additional keyword arguments passed through to internal sample()
            calls, should there be any.

        Returns
        -------
        tuple
            With a single array of shape (n_timepoints,).
        """
        Rt0 = pyro.sample("Rt0", self.Rt0_dist)
        Rt0_trans = self.Rt_transform(Rt0)
        Rt_trans_proc = SimpleRandomWalkProcess(self.Rt_rw_dist)
        Rt_trans_ts = Rt_trans_proc.sample(
            n_timepoints=n_timepoints,
            init=Rt0_trans,
            **kwargs
        )

        Rt = [self.Rt_transform.inv(r) for r in Rt_trans_ts]
        # Convert the list of tensors to a single tensor
        Rt_tensor = torch.stack(Rt)
    
        return (Rt_tensor,)
