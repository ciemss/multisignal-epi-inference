# numpydoc ignore=GL08

import torch
import numpy.testing as testing
from pyrenew.deterministic import (
    DeterministicPMF,
    DeterministicProcess,
    DeterministicVariable,
    NullProcess,
    NullVariable,
)


def test_deterministic():
    """
    Test the DeterministicVariable, DeterministicPMF, and
    DeterministicProcess classes in the deterministic module.
    """

    var1 = DeterministicVariable(
        torch.tensor(
            [
                1,
            ]
        ),
        name="var1",
    )
    var2 = DeterministicPMF(torch.tensor([0.25, 0.25, 0.2, 0.3]), name="var2")
    var3 = DeterministicProcess(torch.tensor([1, 2, 3, 4]), name="var3")
    var4 = NullVariable()
    var5 = NullProcess()

    testing.assert_array_equal(
        var1.sample()[0],
        torch.tensor(
            [
                1,
            ]
        ).numpy(),
    )
    testing.assert_array_equal(
        var2.sample()[0],
        torch.tensor([0.25, 0.25, 0.2, 0.3]).numpy(),
    )
    testing.assert_array_equal(
        var3.sample(duration=5)[0],
        torch.tensor([1, 2, 3, 4, 4]).numpy(),
    )

    testing.assert_array_equal(
        var3.sample(duration=3)[0],
        torch.tensor(
            [
                1,
                2,
                3,
            ]
        ).numpy(),
    )

    testing.assert_equal(var4.sample()[0], None)
    testing.assert_equal(var5.sample(duration=1)[0], None)
