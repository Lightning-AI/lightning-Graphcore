from unittest.mock import Mock

from lightning_utilities.core.imports import package_available

if package_available("lightning"):
    from lightning.pytorch.demos.boring_classes import BoringModel
elif package_available("pytorch_lightning"):
    from pytorch_lightning.demos.boring_classes import BoringModel

from lightning_graphcore.utils import _LightningModuleWrapperBase


def test_wrapper():
    trainer = Mock()
    model = Mock(spec=BoringModel)
    model._trainer = trainer
    wrapped_model = _LightningModuleWrapperBase(model)

    trainer.training = True
    wrapped_model.forward()
    model.training_step.assert_called_once()

    trainer.training = False
    trainer.testing = True
    wrapped_model.forward()
    model.test_step.assert_called_once()

    trainer.testing = False
    trainer.validating = True
    wrapped_model.forward()
    model.validation_step.assert_called_once()

    trainer.validating = False
    trainer.sanity_checking = True
    wrapped_model.forward()
    model.validation_step.assert_called()

    trainer.sanity_checking = False
    trainer.predicting = True
    wrapped_model.forward()
    model.predict_step.assert_called_once()
