"""Root package info."""

import os

from lightning_graphcore.__about__ import *  # noqa: F401, F403
from lightning_graphcore.accelerator import IPUAccelerator
from lightning_graphcore.precision import IPUPrecision
from lightning_graphcore.strategy import IPUStrategy

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)


__all__ = [
    "IPUAccelerator",
    "IPUPrecision",
    "IPUStrategy",
]
