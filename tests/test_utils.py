from unittest.mock import Mock

import pytest
from lightning_utilities.core.imports import package_available

from lightning_graphcore.utils import _LightningModuleWrapperBase

if package_available("lightning"):
    from lightning.pytorch.demos.boring_classes import BoringModel
elif package_available("pytorch_lightning"):
    from pytorch_lightning.demos.boring_classes import BoringModel


@pytest.mark.parametrize(
    ("stage", "step_method"),
    [
        ("training", "training_step"),
        ("validating", "validation_step"),
        ("sanity_checking", "validation_step"),
        ("testing", "test_step"),
        ("predicting", "predict_step"),
    ],
)
def test_wrapper(stage, step_method):
    trainer = Mock()
    model = Mock(spec=BoringModel)
    model._trainer = trainer
    wrapped_model = _LightningModuleWrapperBase(model)

    disabled_stages = {"training", "validating", "sanity_checking", "testing", "predicting"}
    disabled_stages.remove(stage)
    for disabled_stage in disabled_stages:
        setattr(trainer, disabled_stage, False)
    setattr(trainer, stage, True)

    wrapped_model.forward()
    getattr(model, step_method).assert_called_once()
