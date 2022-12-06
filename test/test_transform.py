import torch
import numpy as np
import pytest
from churn import Distillation


y_true = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
y_base_model_softmaxed = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.3, 0.6, 0.1]])
y_base_model = np.array([[11.0, 7.0, -4.0], [7.0, 12.0, 0.5], [1.5, 15.0, 4.0]])


@pytest.mark.parametrize(
    "y_t,y_b,lmbd,expected",
    [
        (y_true, y_base_model, 0.0, torch.Tensor(y_true)),
        (
            y_true,
            y_base_model,
            1.0,
            np.array([[0.98, 0.02, 0.0], [0.01, 0.99, 0.0], [0.0, 1.0, 0.0]]),
        ),
        (
            y_true,
            y_base_model,
            0.5,
            np.array([[0.99, 0.01, 0.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]]),
        ),
    ],
)
def test_distillation_array(y_t, y_b, lmbd, expected):
    transformation = Distillation(lambda_=lmbd)
    y_distilled = transformation.transform(y_t, y_b).numpy().round(2)
    assert np.allclose(y_distilled, expected)


def test_distillation_warning():
    with pytest.warns(
        UserWarning, match="Your y_base_model Tensor might be already softmaxed!"
    ):
        transformation = Distillation(1.0)
        transformation.transform(y_true, y_base_model_softmaxed)
