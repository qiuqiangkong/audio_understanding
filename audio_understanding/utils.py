from __future__ import annotations

import torch
import yaml


def parse_yaml(config_yaml: str) -> dict:
    r"""Parse yaml file."""
    
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


class LinearWarmUp:
    r"""Linear learning rate warm up scheduler.
    """
    def __init__(self, warm_up_steps: int) -> None:
        self.warm_up_steps = warm_up_steps

    def __call__(self, step: int) -> float:
        if step <= self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            return 1.


def pad_or_truncate(x: list, length: int, pad_value: float | str) -> list:
    r"""Pad or truncate sequence."""

    if len(x) >= length:
        return x[: length]
    
    else:
        return x + [pad_value] * (length - len(x))


def remove_padded_columns(ids: torch.LongTensor, pad_token_id: int) -> torch.LongTensor:
    r"""Remove padded columns in a batch to shorten the seuqence length.
    
    E.g., [[1, 3, 2, 0, 0], [7, 0, 0, 0, 0]] -> [[1, 3, 2], [7, 0, 0]]

    Args:
        ids: (b, t)
        pad_token_id: int

    Returns:
        ids: (b, t_new)
    """

    # Indicate whether all ids[:, t] in a batch are pad id
    bool_tensor = torch.all(ids == pad_token_id, dim=0)
    # shape: (t,), e.g., [False, False, False, True, True]

    idx = torch.sum(bool_tensor == False)
    ids = ids[:, 0 : idx]  # (b, t)

    return ids