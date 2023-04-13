"""Root package info."""

import os

from lightning_graphcore.__about__ import *  # noqa: F401, F403
from lightning_graphcore.accelerator import AcceleratorIPU
from lightning_graphcore.precision import PrecisionIPU
from lightning_graphcore.strategy import StrategyIPU

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


__all__ = [
    "AcceleratorIPU",
    "PrecisionIPU",
    "StrategyIPU",
]
