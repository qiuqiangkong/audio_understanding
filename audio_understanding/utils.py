from __future__ import annotations

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