"""
Utility functions for processing arrays.
"""

from typing import NamedTuple
import torch
from torch import Tensor

def pad_to_match(
    x: Tensor,
    y: Tensor,
    fill_value: float = 0.0,
    pad_direction: str = "end",
    fix_y: bool = False,
) -> tuple[Tensor, Tensor]:
    """
    Pad the shorter tensor at the start or end to match the length of the longer tensor.
    """
    x_len = x.size(0)
    y_len = y.size(0)
    pad_size = abs(x_len - y_len)

    if pad_direction == "end":
        pad_width = (0, pad_size)
    elif pad_direction == "start":
        pad_width = (pad_size, 0)
    else:
        raise ValueError(f"pad_direction must be either 'start' or 'end'. Got {pad_direction}.")

    if x_len > y_len:
        if fix_y:
            raise ValueError("Cannot fix y when x is longer than y.")
        y = torch.nn.functional.pad(y, pad_width, "constant", fill_value)

    elif y_len > x_len:
        x = torch.nn.functional.pad(x, pad_width, "constant", fill_value)

    return x, y

def pad_x_to_match_y(
    x: Tensor,
    y: Tensor,
    fill_value: float = 0.0,
    pad_direction: str = "end",
) -> Tensor:
    """
    Pad the `x` tensor at the start or end to match the length of the `y` tensor.
    """
    return pad_to_match(x, y, fill_value=fill_value, pad_direction=pad_direction, fix_y=True)[0]

class PeriodicProcessSample(NamedTuple):
    """
    A container for holding the output from `process.PeriodicProcess.sample()`.
    """
    value: Tensor | None = None

    def __repr__(self) -> str:
        return f"PeriodicProcessSample(value={self.value})"

class PeriodicBroadcaster:
    """
    Broadcast tensors periodically using either repeat or tile.
    """
    def __init__(
        self,
        offset: int,
        period_size: int,
        broadcast_type: str,
    ) -> None:
        self.validate(offset=offset, period_size=period_size, broadcast_type=broadcast_type)
        self.period_size = period_size
        self.offset = offset
        self.broadcast_type = broadcast_type

    @staticmethod
    def validate(offset: int, period_size: int, broadcast_type: str) -> None:
        assert 0 <= offset < period_size, "Offset must be within the range of period_size."
        assert broadcast_type in ["repeat", "tile"], "Broadcast type must be 'repeat' or 'tile'."

    def __call__(self, data: Tensor, n_timepoints: int) -> Tensor:
        """
        Broadcast the data to the given number of timepoints considering the period size and starting point.
        """
        if self.broadcast_type == "repeat":
            repeated_data = data.repeat_interleave(self.period_size)
        elif self.broadcast_type == "tile":
            tiles = -(-n_timepoints // data.size(0))  # Ceiling division
            repeated_data = data.repeat(tiles)

        start_index = self.offset
        end_index = start_index + n_timepoints

        return repeated_data[start_index:end_index]
