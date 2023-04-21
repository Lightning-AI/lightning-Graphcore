# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Callable, Literal, Union, cast

from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.optim import LBFGS, Optimizer
from typing_extensions import get_args

if package_available("lightning"):
    from lightning.fabric.utilities.types import Optimizable
    from lightning.pytorch import LightningModule
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin
    from lightning.pytorch.utilities import GradClipAlgorithmType
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
    from lightning.pytorch.utilities.model_helpers import is_overridden
    from lightning.pytorch.utilities.rank_zero import WarningCache
elif package_available("pytorch_lightning"):
    from lightning_fabric.utilities.types import Optimizable
    from pytorch_lightning import LightningModule
    from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
    from pytorch_lightning.utilities import GradClipAlgorithmType
    from pytorch_lightning.utilities.exceptions import MisconfigurationException
    from pytorch_lightning.utilities.model_helpers import is_overridden
    from pytorch_lightning.utilities.rank_zero import WarningCache
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

warning_cache = WarningCache()

_PRECISION_INPUT = Literal["32-true", "16-mixed"]


class IPUPrecision(PrecisionPlugin):
    """Precision plugin for IPU integration.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    Raises:
        ValueError:
            If the precision is neither 16-mixed nor 32-true.
    """

    def __init__(self, precision: Literal["32-true", "16-mixed"]) -> None:
        supported_precision = get_args(_PRECISION_INPUT)
        if precision not in supported_precision:
            raise ValueError(
                f"`Trainer(accelerator='ipu', precision={precision!r})` is not supported."
                f" `precision` must be one of: {supported_precision}."
            )
        self.precision = cast(_PRECISION_INPUT, str(precision))

    def backward(
        self,
        tensor: Tensor,
        model: LightningModule,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since IPUs handle"
                " the backward logic internally."
            )

    def optimizer_step(
        self,
        optimizer: Optimizable,
        model: LightningModule,
        closure: Callable[[], Any],
        **kwargs: Any,
    ) -> Any:
        """IPUs handle the optimizer step internally."""
        if isinstance(optimizer, LBFGS):
            raise MisconfigurationException("IPUs and the LBFGS optimizer are not compatible.")
        closure_result = closure()
        self._after_closure(model, optimizer)
        skipped_backward = closure_result is None
        # in manual optimization, the closure does not return a value
        if model.automatic_optimization and skipped_backward:
            # we lack coverage here and IPUs are (currently) limited - something to explore if there's demand
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not implemented for IPUs."
                " Please, open an issue in `https://github.com/Lightning-AI/lightning/issues`"
                " requesting this feature."
            )
        return closure_result

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[int, float] = 0.0,
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        if clip_val <= 0:
            return
        raise MisconfigurationException("IPUs currently do not support clipping gradients.")
