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
from typing import Any, Union

import torch
from lightning_utilities.core.imports import package_available

if package_available("lightning"):
    from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
    from lightning.pytorch import LightningModule
    from lightning.pytorch.overrides.base import _LightningPrecisionModuleWrapperBase
elif package_available("pytorch_lightning"):
    from pytorch_lightning import LightningModule
    from pytorch_lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
    from pytorch_lightning.overrides.base import _LightningPrecisionModuleWrapperBase


class _LightningModuleWrapperBase(_DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(self, forward_module: Union[LightningModule, _LightningPrecisionModuleWrapperBase]) -> None:
        """Wrap the user's LightningModule and redirect the forward call to the appropriate `*_step()` methods.

        Inheriting classes may also modify the inputs or outputs of forward.

        Args:
            forward_module: The module to wrap. If it's not a LightningModule, it must have an attribute ``.module``
                pointing to a LightningModule reference.
        """
        super().__init__()
        if not isinstance(forward_module, LightningModule) and (
            not isinstance(getattr(forward_module, "module", None), LightningModule)
        ):
            raise ValueError(
                "`forward_module` must be a `LightningModule` instance or have an attribute `.module` pointing to one,"
                f" got: {forward_module.__class__.__qualname__}"
            )
        self._forward_module = forward_module

    @property
    def lightning_module(self) -> LightningModule:
        if isinstance(self._forward_module, LightningModule):
            return self._forward_module
        return self._forward_module.module

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        pl_module = self.lightning_module
        trainer = pl_module._trainer

        if trainer is not None:
            if trainer.training:
                return self._forward_module.training_step(*inputs, **kwargs)
            if trainer.testing:
                return self._forward_module.test_step(*inputs, **kwargs)
            if trainer.sanity_checking or trainer.validating:
                return self._forward_module.validation_step(*inputs, **kwargs)
            if trainer.predicting:
                return self._forward_module.predict_step(*inputs, **kwargs)
        return self._forward_module(*inputs, **kwargs)
