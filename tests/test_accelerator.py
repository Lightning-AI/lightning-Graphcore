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
import os
from unittest import mock

import poptorch
import pytest
import torch
from lightning_utilities.core.imports import package_available
from torch.utils.data import DistributedSampler

if package_available("lightning"):
    from lightning.pytorch import Callback, Trainer, seed_everything
    from lightning.pytorch.core.module import LightningModule
    from lightning.pytorch.demos.boring_classes import BoringModel
    from lightning.pytorch.trainer.states import RunningStage, TrainerFn
    from lightning.pytorch.utilities.exceptions import MisconfigurationException

elif package_available("pytorch_lightning"):
    from pytorch_lightning import Callback, Trainer, seed_everything
    from pytorch_lightning.core.module import LightningModule
    from pytorch_lightning.demos.boring_classes import BoringModel
    from pytorch_lightning.trainer.states import RunningStage, TrainerFn
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_graphcore import IPUStrategy
from lightning_graphcore.accelerator import IPUAccelerator
from lightning_graphcore.precision import IPUPrecision
from tests.helpers import ClassifDataModule, IPUClassificationModel, IPUModel


def test_auto_device_count():
    assert IPUAccelerator.auto_device_count() == 4


@mock.patch("lightning_graphcore.accelerator.IPUAccelerator.is_available", return_value=False)
def test_fail_if_no_ipus(_, tmpdir):  # noqa: PT019
    with pytest.raises(
        MisconfigurationException,
        match="`IPUAccelerator` can not run on your system since the accelerator is not available.",
    ):
        Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=1)


@pytest.mark.xfail()  # todo
def test_accelerator_selected(tmpdir):
    assert IPUAccelerator.is_available()
    trainer = Trainer(default_root_dir=tmpdir, accelerator="ipu", devices=1)
    assert isinstance(trainer.accelerator, IPUAccelerator)


def test_warning_if_ipus_not_used():
    with pytest.warns(UserWarning, match="IPU available but not used. Set `accelerator` and `devices`"):
        Trainer(accelerator="cpu")


def test_no_warning_strategy(tmpdir):
    with pytest.warns(None) as record:
        Trainer(default_root_dir=tmpdir, max_epochs=1, strategy=IPUStrategy(training_opts=poptorch.Options()))
    assert len(record) == 0


@pytest.mark.parametrize(
    "devices",
    [1, 4],
)
def test_all_stages(tmpdir, devices):
    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy=IPUStrategy(), devices=devices)
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@pytest.mark.parametrize(
    "devices",
    [
        1,
        pytest.param(
            4,
            marks=pytest.mark.xfail(  # fixme
                AssertionError, reason="Invalid batch dimension: In the input torch.Size([1, 32]), ..."
            ),
        ),
    ],
)
def test_inference_only(tmpdir, devices):
    model = IPUModel()

    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, strategy=IPUStrategy(accelerator=IPUAccelerator()), devices=devices
    )
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@pytest.mark.xfail(reason="TODO: issues with latest poptorch")
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = IPUClassificationModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator=IPUAccelerator(), devices=2)

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]["val_acc"] > 0.7

    # test
    result = trainer.test(model, datamodule=dm)
    assert dm.trainer is not None
    test_result = result[0]["test_acc"]
    assert test_result > 0.6

    # test saved model
    model_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(model_path)

    model = IPUClassificationModel.load_from_checkpoint(model_path)

    trainer = Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=2)

    result = trainer.test(model, datamodule=dm)
    saved_result = result[0]["test_acc"]
    assert saved_result == test_result


def test_half_precision(tmpdir):
    class TestCallback(Callback):
        def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            assert trainer.precision == "16-mixed"
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        strategy=IPUStrategy(accelerator=IPUAccelerator(), precision_plugin=IPUPrecision("16-mixed")),
        devices=1,
        callbacks=TestCallback(),
    )
    assert isinstance(trainer.strategy.precision_plugin, IPUPrecision)
    assert trainer.strategy.precision_plugin.precision == "16-mixed"
    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_pure_half_precision(tmpdir):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert trainer.strategy.precision_plugin.precision == "16-mixed"
            for param in trainer.strategy.model.parameters():
                assert param.dtype == torch.float16
            raise SystemExit

    model = IPUModel()
    model = model.half()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        strategy=IPUStrategy(accelerator=IPUAccelerator(), precision_plugin=IPUPrecision("16-mixed")),
        devices=1,
        callbacks=TestCallback(),
    )

    assert isinstance(trainer.strategy, IPUStrategy)
    assert isinstance(trainer.strategy.precision_plugin, IPUPrecision)
    assert trainer.strategy.precision_plugin.precision == "16-mixed"

    changed_dtypes = [torch.float, torch.float64]
    data = [torch.zeros((1), dtype=dtype) for dtype in changed_dtypes]
    new_data = trainer.strategy.batch_to_device(data)
    assert all(val.dtype is torch.half for val in new_data), "".join(
        [f"{dtype}: {val.dtype}" for dtype, val in zip(changed_dtypes, new_data)]
    )

    not_changed_dtypes = [torch.uint8, torch.int8, torch.int32, torch.int64]
    data = [torch.zeros((1), dtype=dtype) for dtype in not_changed_dtypes]
    new_data = trainer.strategy.batch_to_device(data)
    assert all(val.dtype is dtype for val, dtype in zip(new_data, not_changed_dtypes)), "".join(
        [f"{dtype}: {val.dtype}" for dtype, val in zip(not_changed_dtypes, new_data)]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_device_iterations_ipu_strategy(tmpdir):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            assert trainer.strategy.device_iterations == 2
            # assert device iterations has been set correctly within the poptorch options
            poptorch_model = trainer.strategy.poptorch_models[RunningStage.TRAINING]
            assert poptorch_model._options.toDict()["device_iterations"] == 2
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=IPUAccelerator(),
        devices=1,
        strategy=IPUStrategy(device_iterations=2),
        callbacks=TestCallback(),
    )
    assert isinstance(trainer.strategy, IPUStrategy)
    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_accumulated_batches(tmpdir):
    class TestCallback(Callback):
        def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
            # assert poptorch option have been set correctly
            poptorch_model = trainer.strategy.poptorch_models[RunningStage.TRAINING]
            assert poptorch_model._options.Training.toDict()["gradient_accumulation"] == 2
            raise SystemExit

    model = IPUModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        strategy=IPUStrategy(accelerator=IPUAccelerator()),
        devices=1,
        accumulate_grad_batches=2,
        callbacks=TestCallback(),
    )
    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage."""

    class StageModel(IPUModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            return (loss - loss) + torch.tensor(1)

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            return (loss - loss) + torch.tensor(2)

        def test_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            return (loss - loss) + torch.tensor(3)

        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs["loss"].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs.item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert outputs.item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, *_) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator=IPUAccelerator(), devices=1, callbacks=TestCallback()
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model, model.test_dataloader())


@pytest.mark.skip()  # todo: did not raise exception
def test_clip_gradients_fails(tmpdir):
    model = IPUModel()
    trainer = Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=1, gradient_clip_val=10)
    with pytest.raises(MisconfigurationException, match="IPUs currently do not support clipping gradients."):
        trainer.fit(model)


@pytest.mark.xfail(RuntimeError, reason="element 0 of tensors does not require grad and does not have ...")  # todo
def test_autoreport(tmpdir):
    """Ensure autoreport dumps to a file."""
    model = IPUModel()
    autoreport_path = os.path.join(tmpdir, "report/")
    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=IPUAccelerator(),
        devices=1,
        fast_dev_run=True,
        strategy=IPUStrategy(autoreport=True, autoreport_dir=autoreport_path),
    )
    trainer.fit(model)
    assert os.path.exists(autoreport_path)
    assert os.path.isfile(autoreport_path + "training/profile.pop")


@pytest.mark.xfail(RuntimeError, reason="element 0 of tensors does not require grad and does not have ...")  # todo
def test_manual_poptorch_dataloader(tmpdir):
    model_options = poptorch.Options()

    class IPUTestModel(IPUModel):
        def train_dataloader(self):
            dataloader = super().train_dataloader()
            # save to instance to compare the reference later
            self.poptorch_dataloader = poptorch.DataLoader(model_options, dataloader.dataset, drop_last=True)
            return self.poptorch_dataloader

    model = IPUTestModel()
    other_options = poptorch.Options()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator=IPUAccelerator(),
        devices=2,
        strategy=IPUStrategy(training_opts=other_options),
    )
    trainer.fit(model)

    assert isinstance(trainer.strategy, IPUStrategy)
    assert trainer.strategy.training_opts is other_options
    dataloader = trainer.train_dataloader
    assert dataloader is model.poptorch_dataloader  # exact object, was not recreated
    # dataloader uses the options in the model, not the strategy
    assert dataloader.options is model_options
    assert dataloader.options is not other_options
    assert dataloader.drop_last  # was kept


@pytest.mark.xfail(RuntimeError, reason="element 0 of tensors does not require grad and does not have ...")  # todo
def test_manual_poptorch_opts(tmpdir):
    """Ensure if the user passes manual poptorch Options, we run with the correct object."""
    model = IPUModel()
    inference_opts = poptorch.Options()
    training_opts = poptorch.Options()

    trainer = Trainer(
        default_root_dir=tmpdir,
        accelerator=IPUAccelerator(),
        devices=2,
        fast_dev_run=True,
        strategy=IPUStrategy(inference_opts=inference_opts, training_opts=training_opts),
    )
    trainer.fit(model)

    assert isinstance(trainer.strategy, IPUStrategy)
    assert trainer.strategy.training_opts == training_opts
    assert trainer.strategy.inference_opts == inference_opts

    dataloader = trainer.train_dataloader
    assert isinstance(dataloader, poptorch.DataLoader)
    assert dataloader.options == training_opts
    assert trainer.num_devices > 1  # testing this only makes sense in a distributed setting
    assert not isinstance(dataloader.sampler, DistributedSampler)


def test_manual_poptorch_opts_custom(tmpdir):
    """Ensure if the user passes manual poptorch Options with custom parameters set.

    We respect them in our poptorch options and the dataloaders.
    """
    model = IPUModel()
    training_opts = poptorch.Options()
    training_opts.deviceIterations(8)
    training_opts.replicationFactor(2)
    training_opts.Training.gradientAccumulation(2)

    inference_opts = poptorch.Options()
    inference_opts.deviceIterations(16)
    inference_opts.replicationFactor(1)
    inference_opts.Training.gradientAccumulation(1)

    class TestCallback(Callback):
        def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
            # ensure dataloaders were correctly set up during training.
            strategy = trainer.strategy
            assert isinstance(strategy, IPUStrategy)
            assert strategy.training_opts.replication_factor == 2
            assert strategy.inference_opts.replication_factor == 1

            val_dataloader = trainer.val_dataloaders
            train_dataloader = trainer.train_dataloader
            assert isinstance(val_dataloader, poptorch.DataLoader)
            assert isinstance(train_dataloader, poptorch.DataLoader)
            assert train_dataloader.options.replication_factor == 2
            assert val_dataloader.options.replication_factor == 1

    strategy = IPUStrategy(inference_opts=inference_opts, training_opts=training_opts)
    # ensure we default to the training options replication factor
    assert strategy.replication_factor == 2
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, strategy=strategy, callbacks=TestCallback())
    trainer.fit(model)

    strategy = trainer.strategy
    assert isinstance(strategy, IPUStrategy)

    training_opts = strategy.training_opts
    assert training_opts.device_iterations == 8
    assert training_opts.replication_factor == 2
    assert training_opts.Training.gradient_accumulation == 2

    inference_opts = strategy.inference_opts
    assert inference_opts.device_iterations == 16
    assert inference_opts.replication_factor == 1
    assert inference_opts.Training.gradient_accumulation == 1


def test_replication_factor(tmpdir):
    """Ensure if the user passes manual poptorch Options with custom parameters set.

    We set them correctly in the dataloaders.
    """
    strategy = IPUStrategy()
    trainer = Trainer(
        accelerator=IPUAccelerator(), devices=2, default_root_dir=tmpdir, fast_dev_run=True, strategy=strategy
    )
    assert isinstance(trainer.accelerator, IPUAccelerator)
    assert trainer.num_devices == 2
    assert trainer.strategy.replication_factor == 2

    model = BoringModel()
    training_opts = poptorch.Options()
    inference_opts = poptorch.Options()
    training_opts.replicationFactor(8)
    inference_opts.replicationFactor(7)
    strategy = IPUStrategy(inference_opts=inference_opts, training_opts=training_opts)

    trainer = Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=1, strategy=strategy)
    trainer.optimizers = model.configure_optimizers()[0]
    strategy._lightning_module = model
    model.trainer = trainer
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.setup(trainer)

    trainer.state.stage = RunningStage.TRAINING
    assert trainer.strategy.replication_factor == 8
    trainer.state.stage = RunningStage.VALIDATING
    assert trainer.strategy.replication_factor == 7

    for fn, stage in (
        (TrainerFn.VALIDATING, RunningStage.VALIDATING),
        (TrainerFn.TESTING, RunningStage.TESTING),
        (TrainerFn.PREDICTING, RunningStage.PREDICTING),
    ):
        trainer.state.fn = fn
        trainer.state.stage = stage
        trainer.strategy.setup(trainer)
        assert trainer.strategy.replication_factor == 7


@pytest.mark.xfail()  # todo: asser strategy is COU
def test_default_opts(tmpdir):
    """Ensure default opts are set correctly in the IPUStrategy."""
    model = IPUModel()

    trainer = Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=1, fast_dev_run=True)
    trainer.fit(model)
    assert isinstance(trainer.strategy, IPUStrategy)
    inference_opts = trainer.strategy.inference_opts
    training_opts = trainer.strategy.training_opts
    for opts in (inference_opts, training_opts):
        assert isinstance(opts, poptorch.Options)
        assert opts.Training.gradient_accumulation == 1
        assert opts.device_iterations == 1
        assert opts.replication_factor == 1


@pytest.mark.skip()  # todo: did not raise exception
def test_multi_optimizers_fails(tmpdir):
    """Ensure if there are multiple optimizers, we throw an exception."""

    class TestModel(IPUModel):
        def configure_optimizers(self):
            return [torch.optim.Adam(self.parameters()), torch.optim.Adam(self.parameters())]

    model = TestModel()
    # Must switch to manual optimization mode, otherwise we would get a different error
    # (multiple optimizers only supported with manual optimization)
    model.automatic_optimization = False

    trainer = Trainer(default_root_dir=tmpdir, accelerator=IPUAccelerator(), devices=1)
    with pytest.raises(MisconfigurationException, match="IPUs currently only support one optimizer."):
        trainer.fit(model)


def test_precision_plugin():
    """Ensure precision plugin value is set correctly."""
    plugin = IPUPrecision(precision="16-mixed")
    assert plugin.precision == "16-mixed"


def test_accelerator_ipu():
    trainer = Trainer(accelerator=IPUAccelerator(), devices=1)
    assert isinstance(trainer.accelerator, IPUAccelerator)

    trainer = Trainer(accelerator=IPUAccelerator())
    assert isinstance(trainer.accelerator, IPUAccelerator)

    # TODO
    # trainer = Trainer(accelerator="auto", devices=8)
    # assert isinstance(trainer.accelerator, IPUAccelerator)


def test_accelerator_ipu_with_devices():
    trainer = Trainer(strategy=IPUStrategy(accelerator=IPUAccelerator()), devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert isinstance(trainer.accelerator, IPUAccelerator)
    assert trainer.num_devices == 8


@pytest.mark.xfail(AssertionError, reason="not implemented on PL side")
def test_accelerator_auto_with_devices_ipu():
    trainer = Trainer(accelerator="auto", devices=8)
    assert isinstance(trainer.accelerator, IPUAccelerator)
    assert trainer.num_devices == 8


def test_strategy_choice_ipu_strategy():
    trainer = Trainer(strategy=IPUStrategy(), accelerator=IPUAccelerator(), devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)


def test_device_type_when_ipu_strategy_passed():
    trainer = Trainer(strategy=IPUStrategy(), accelerator=IPUAccelerator(), devices=8)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert isinstance(trainer.accelerator, IPUAccelerator)


def test_poptorch_models_at_different_stages(tmpdir):
    strategy = IPUStrategy()
    trainer = Trainer(default_root_dir=tmpdir, strategy=strategy, accelerator=IPUAccelerator(), devices=8)
    model = BoringModel()
    model.trainer = trainer
    strategy._lightning_module = model

    trainer.optimizers = model.configure_optimizers()[0]
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.setup(trainer)
    assert list(trainer.strategy.poptorch_models) == [
        RunningStage.TRAINING,
        RunningStage.VALIDATING,
        RunningStage.SANITY_CHECKING,
    ]

    for fn, stage in (
        (TrainerFn.VALIDATING, RunningStage.VALIDATING),
        (TrainerFn.TESTING, RunningStage.TESTING),
        (TrainerFn.PREDICTING, RunningStage.PREDICTING),
    ):
        trainer.state.fn = fn
        trainer.state.stage = stage
        trainer.strategy.setup(trainer)
        assert list(trainer.strategy.poptorch_models) == [stage]


@pytest.mark.xfail(AssertionError, reason="not implemented on PL side")
def test_devices_auto_choice_ipu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.num_devices == 4
    assert isinstance(trainer.accelerator, IPUAccelerator)


@pytest.mark.parametrize("trainer_kwargs", [{"accelerator": "ipu"}])
@pytest.mark.parametrize("hook", ["transfer_batch_to_device", "on_after_batch_transfer"])
def test_raise_exception_with_batch_transfer_hooks(monkeypatch, hook, trainer_kwargs, tmpdir):
    """Test that an exception is raised when overriding batch_transfer_hooks."""

    def custom_method(self, batch, *_, **__):
        return batch + 1

    trainer = Trainer(default_root_dir=tmpdir, accelerator="ipu")

    model = BoringModel()
    setattr(model, hook, custom_method)

    match_pattern = rf"Overriding `{hook}` is not .* with IPUs"
    with pytest.raises(MisconfigurationException, match=match_pattern):
        trainer.fit(model)


def test_unsupported_ipu_choice(monkeypatch):
    with pytest.raises(ValueError, match=r"accelerator='ipu', precision='bf16-mixed'\)` is not supported"):
        Trainer(accelerator="ipu", precision="bf16-mixed")
    with pytest.raises(ValueError, match=r"accelerator='ipu', precision='64-true'\)` is not supported"):
        Trainer(accelerator="ipu", precision="64-true")


@pytest.mark.xfail()  # todo
def test_num_stepping_batches_with_ipu(monkeypatch):
    """Test stepping batches with IPU training which acts like DP."""
    import lightning.pytorch.strategies.ipu as ipu

    monkeypatch.setattr(ipu, "_IPU_AVAILABLE", True)
    trainer = Trainer(accelerator="ipu", devices=2, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert isinstance(trainer.strategy, IPUStrategy)
    assert trainer.estimated_stepping_batches == 64
