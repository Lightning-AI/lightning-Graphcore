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
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

if package_available("lightning"):
    import lightning.pytorch as pl
    from lightning.fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning.fabric.utilities.cloud_io import get_filesystem
    from lightning.pytorch import Trainer
    from lightning.pytorch.accelerators import Accelerator
    from lightning.pytorch.plugins.precision import PrecisionPlugin
    from lightning.pytorch.strategies.parallel import ParallelStrategy
    from lightning.pytorch.strategies.strategy import TBroadcast
    from lightning.pytorch.strategies.utils import _fp_to_half
    from lightning.pytorch.trainer.states import RunningStage, TrainerFn
    from lightning.pytorch.utilities import rank_zero_warn
    from lightning.pytorch.utilities.data import _get_dataloader_init_args_and_kwargs, _reinstantiate_wrapped_cls
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
    from lightning.pytorch.utilities.model_helpers import is_overridden
    from lightning.pytorch.utilities.types import STEP_OUTPUT
elif package_available("pytorch_lightning"):
    import pytorch_lightning as pl
    from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
    from lightning_fabric.utilities.cloud_io import get_filesystem
    from pytorch_lightning import Trainer
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.plugins.precision import PrecisionPlugin
    from pytorch_lightning.strategies.parallel import ParallelStrategy
    from pytorch_lightning.strategies.strategy import TBroadcast
    from pytorch_lightning.strategies.utils import _fp_to_half
    from pytorch_lightning.trainer.states import RunningStage, TrainerFn
    from pytorch_lightning.utilities import rank_zero_warn
    from pytorch_lightning.utilities.data import _get_dataloader_init_args_and_kwargs, _reinstantiate_wrapped_cls
    from pytorch_lightning.utilities.exceptions import MisconfigurationException
    from pytorch_lightning.utilities.model_helpers import is_overridden
    from pytorch_lightning.utilities.types import STEP_OUTPUT
else:
    raise ModuleNotFoundError("You are missing `lightning` or `pytorch-lightning` package, please install it.")

from lightning_graphcore.accelerator import _IPU_AVAILABLE, _POPTORCH_AVAILABLE
from lightning_graphcore.utils import _LightningModuleWrapperBase

if _POPTORCH_AVAILABLE:
    import poptorch
else:
    poptorch = None


class IPUStrategy(ParallelStrategy):
    """Plugin for training on IPU devices.

    Args:
        device_iterations: Number of iterations to run on device at once before returning to host.
            This can be used as an optimization to speed up training.
            https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html
        autoreport: Enable auto-reporting for IPUs using PopVision
            https://docs.graphcore.ai/projects/graphcore-popvision-user-guide/en/latest/graph/graph.html
        autoreport_dir: Optional directory to store autoReport output.
        training_opts: Optional ``poptorch.Options`` to override the default created options for training.
        inference_opts: Optional ``poptorch.Options`` to override the default
            created options for validation/testing and predicting.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.
    """

    strategy_name = "ipu_strategy"

    def __init__(
        self,
        accelerator: Optional[Accelerator] = None,
        device_iterations: int = 1,
        autoreport: bool = False,
        autoreport_dir: Optional[str] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        training_opts: Optional["poptorch.Options"] = None,
        inference_opts: Optional["poptorch.Options"] = None,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        if not _IPU_AVAILABLE:
            raise MisconfigurationException(
                "The IPU Accelerator requires IPU devices to run. "
                "Learn more or get started with IPUs at https://www.graphcore.ai/getstarted"
            )

        self.device_iterations = device_iterations
        self.autoreport = autoreport
        self.autoreport_dir = autoreport_dir
        self.poptorch_models: Dict[RunningStage, "poptorch.PoplarExecutor"] = {}
        self._training_opts = training_opts
        self._inference_opts = inference_opts

        if self.autoreport:
            options: Dict[str, Any] = {"autoReport.all": self.autoreport}
            if self.autoreport_dir:
                self._fs = get_filesystem(str(self.autoreport_dir))
                self._fs.makedirs(self.autoreport_dir, exist_ok=True)
                options["autoReport.directory"] = self.autoreport_dir
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(options)

        self._update_dataloader_original: Optional[Callable] = None
        self._optimizer_zero_grad_original: Optional[Callable] = None

    def setup(self, trainer: Trainer) -> None:
        # patch the dataloader creation function with the custom `poptorch.DataLoader`.
        # this violates the intended control flow for the plugins, but since this is experimental, we have chosen
        # to use the simpler solution before adding abstractions to override the `DataLoader` class
        self._update_dataloader_original = pl.trainer.connectors.data_connector._update_dataloader
        pl.trainer.connectors.data_connector._update_dataloader = self._convert_to_poptorch_loader

        super().setup(trainer)

        assert self.lightning_module is not None

        # disable the `optimizer_zero_grad` function by setting it to `None`.
        # this is because the IPU zeros the gradients internally
        self._optimizer_zero_grad_original = self.lightning_module.optimizer_zero_grad
        self._disable_zero_grad()

        self.model = _LightningModuleWrapperBase(self.lightning_module)

        # reset the backup
        self.poptorch_models = {}

        # Separate models are instantiated for different stages, but they share the same weights on host.
        # When validation/test models are run, weights are synced first.
        trainer_fn = self.lightning_module.trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            # Create model for training and validation which will run on fit
            training_opts = self.training_opts
            inference_opts = self.inference_opts
            optimizer = self.lightning_module.trainer.optimizers[0]
            model = poptorch.trainingModel(model=self.model, options=training_opts, optimizer=optimizer)
            self.poptorch_models[RunningStage.TRAINING] = model

            if self.lightning_module.trainer.enable_validation:
                model = poptorch.inferenceModel(model=self.model, options=inference_opts)
                self.poptorch_models[RunningStage.VALIDATING] = model
                if self.lightning_module.trainer.num_sanity_val_steps > 0:
                    self.poptorch_models[RunningStage.SANITY_CHECKING] = model
        elif trainer_fn == TrainerFn.VALIDATING:
            model = poptorch.inferenceModel(model=self.model, options=self.inference_opts)
            self.poptorch_models[RunningStage.VALIDATING] = model
        elif trainer_fn == TrainerFn.TESTING:
            model = poptorch.inferenceModel(model=self.model, options=self.inference_opts)
            self.poptorch_models[RunningStage.TESTING] = model
        elif trainer_fn == TrainerFn.PREDICTING:
            model = poptorch.inferenceModel(model=self.model, options=self.inference_opts)
            self.poptorch_models[RunningStage.PREDICTING] = model

    def setup_optimizers(self, trainer: Trainer) -> None:
        super().setup_optimizers(trainer)

        if len(self.optimizers) > 1:
            raise MisconfigurationException("IPUs currently only support one optimizer.")

    @property
    def replication_factor(self) -> int:
        if not self.lightning_module or not self.poptorch_models:
            # The plugin has been passed in by the user and has not been connected to the Trainer.
            # Check if the user has passed in custom poptorch.Options to infer number of IPUs being used.
            # In this scenario we prioritize the training options.
            if self._training_opts:
                return self._training_opts.replication_factor
            if self._inference_opts:
                return self._inference_opts.replication_factor

            assert self.parallel_devices
            return len(self.parallel_devices)
        stage = self.lightning_module.trainer.state.stage
        assert stage is not None
        return self.poptorch_models[stage]._options.toDict()["replication_factor"]

    def _create_opts(self, training: bool) -> "poptorch.Options":
        assert self.lightning_module is not None
        opts = poptorch.Options()
        opts.deviceIterations(self.device_iterations)
        opts.replicationFactor(self.replication_factor)
        gradient_accumulation = self.lightning_module.trainer.accumulate_grad_batches if training else 1
        opts.Training.gradientAccumulation(gradient_accumulation)

        if os.environ.get("PL_GLOBAL_SEED"):
            opts.randomSeed(int(os.environ["PL_GLOBAL_SEED"]))
        return opts

    @property
    def training_opts(self) -> "poptorch.Options":
        if self._training_opts is None:
            self._training_opts = self._create_opts(training=True)
        return self._training_opts

    @property
    def inference_opts(self) -> "poptorch.Options":
        if self._inference_opts is None:
            self._inference_opts = self._create_opts(training=False)
        return self._inference_opts

    def _convert_to_poptorch_loader(
        self, dataloader: DataLoader, sampler: Union[Sampler, Iterable], mode: Optional[RunningStage] = None
    ) -> "poptorch.DataLoader":
        if isinstance(dataloader, poptorch.DataLoader):
            # the user is returning the `poptorch.DataLoader` directly, don't change anything.
            return dataloader

        dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(
            dataloader, sampler, mode, self.replication_factor > 1
        )
        opts = self.training_opts if mode == RunningStage.TRAINING else self.inference_opts
        dataloader = _reinstantiate_wrapped_cls(
            dataloader, opts, *dl_args, explicit_cls=poptorch.DataLoader, **dl_kwargs
        )
        return dataloader

    @property
    def _n_replicate(self) -> int:
        assert self.lightning_module is not None
        opts = self.training_opts if self.lightning_module.training else self.inference_opts
        accumulate_grad_batches = opts.Training.gradient_accumulation
        device_iterations = opts.device_iterations
        replication_factor = opts.replication_factor
        return replication_factor * device_iterations * accumulate_grad_batches

    def _prepare_input(self, args: Any) -> Any:
        def to_tuple(x: Any) -> Tuple:
            return tuple(x)

        def to_tensor(x: Any) -> Tensor:
            return torch.tensor(x).unsqueeze(0).repeat(self._n_replicate)

        args = apply_to_collection(args, dtype=list, function=to_tuple)
        return apply_to_collection(args, dtype=(int, float), function=to_tensor)

    def batch_to_device(self, batch: Any, device: Optional[torch.device] = None, dataloader_idx: int = 0) -> Any:
        # This override is necessary because the cast must occur before the data
        # is moved to the device to prevent wasteful host->device copies.
        # We don't call `super().batch_to_device` because `data.to(device)` is not
        # currently necessary for IPUs. The movement of data from host<->IPU is
        # currently handled by PopTorch.
        return apply_to_collection(batch, Tensor, function=_fp_to_half, precision=self.precision_plugin.precision)

    def _disable_zero_grad(self) -> None:
        lightning_module = self.lightning_module
        assert lightning_module is not None
        if is_overridden("optimizer_zero_grad", lightning_module):
            assert lightning_module is not None  # `is_overridden` returns False otherwise
            rank_zero_warn(
                "You have overridden the `LightningModule.optimizer_zero_grad` hook but it will be ignored since"
                " IPUs handle the zeroing of gradients internally."
            )
        lightning_module.optimizer_zero_grad = None

    def _step(self, stage: RunningStage, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        args = self._prepare_input(args)
        poptorch_model = self.poptorch_models[stage]
        with pl.core.module._jit_is_scripting():
            return poptorch_model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self._step(RunningStage.TRAINING, *args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self._step(RunningStage.VALIDATING, *args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self._step(RunningStage.TESTING, *args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self._step(RunningStage.PREDICTING, *args, **kwargs)

    def teardown(self) -> None:
        if self._update_dataloader_original is not None:
            # undo dataloader patching
            pl.trainer.connectors.data_connector._update_dataloader = self._update_dataloader_original

        assert self.lightning_module is not None
        if self._optimizer_zero_grad_original is not None:
            # re-enable `optimizer_zero_grad`
            self.lightning_module.optimizer_zero_grad = self._optimizer_zero_grad_original

        for model in self.poptorch_models.values():
            model.destroy()

        super().teardown()

    def _compiled(self, model: Any) -> bool:
        # Required to ensure we only attach compiled models, as they are compiled lazily.
        return model._executable is not None

    def _detach_models(self) -> None:
        """Detaches all stage specific models from IPU devices."""
        for k, model in self.poptorch_models.items():
            if self._compiled(model) and model.isAttachedToDevice():
                model.detachFromDevice()

    def _load_model(self, stage: RunningStage) -> None:
        """Load the stage specific accelerator model onto device if compiled and not attached to IPU devices.

        Args:
            stage: The stage to load
        """
        self._detach_models()
        model = self.poptorch_models[stage]
        if self._compiled(model) and not model.isAttachedToDevice():
            model.attachToDevice()

    def on_train_start(self) -> None:
        self._load_model(RunningStage.TRAINING)

    def on_validation_start(self) -> None:
        self._load_model(RunningStage.VALIDATING)

    def on_test_start(self) -> None:
        self._load_model(RunningStage.TESTING)

    def on_predict_start(self) -> None:
        self._load_model(RunningStage.PREDICTING)

    def on_train_end(self) -> None:
        self._detach_models()

    def on_validation_end(self) -> None:
        self._detach_models()

    def on_test_end(self) -> None:
        self._detach_models()

    def on_predict_end(self) -> None:
        self._detach_models()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        # Updates optimizer stats if LR scheduler modified the optimizer state
        optimizer = self.optimizers[0]
        self.poptorch_models[RunningStage.TRAINING].setOptimizer(optimizer)

    @property
    def root_device(self) -> torch.device:  # type: ignore[empty-body]
        # TODO: this should return `self.parallel_devices[self.local_rank]`
        pass

    def model_to_device(self) -> None:
        pass

    @property
    def is_global_zero(self) -> bool:
        return True

    def reduce(self, tensor: Union[Tensor, Any], *args: Any, **kwargs: Any) -> Union[Tensor, Any]:
        return tensor

    def barrier(self, name: Optional[str] = None) -> None:
        pass

    def all_gather(self, tensor: Tensor, group: Optional[Any] = None, sync_grads: bool = False) -> Tensor:
        return tensor

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        return obj

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
