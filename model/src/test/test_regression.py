import torch
import pyro
import pyro.distributions as dist
from pyrenew.regression import GLMPrediction
from pyrenew.transformation import IdentityTransform
import pytest

def test_glm_prediction():
    """
    Test generalized linear model prediction functionality.
    """
    intercept_custom_suffix = "_523sdgahbf"
    coefficient_custom_suffix = "_gad23562g%"
    fixed_predictor_values = torch.tensor([[2, 0.5, -7, 3], [1, 20, -15, 0]], dtype=torch.float32)

    glm_pred = GLMPrediction(
        "test_GLM_prediction",
        fixed_predictor_values=fixed_predictor_values,
        intercept_prior=dist.Normal(0, 1.5),
        coefficient_priors=dist.Normal(0, 0.5).expand([4]),
        transform=IdentityTransform(),
        intercept_suffix=intercept_custom_suffix,
        coefficient_suffix=coefficient_custom_suffix,
    )

    # Check that transform is identity if not set explicitly
    assert isinstance(glm_pred.transform, IdentityTransform)

    # Deterministic predictions should work as matrix algebra
    fixed_pred_coeff = torch.tensor([1, 35235, -5232.2532, 0], dtype=torch.float32)
    fixed_pred_intercept = torch.tensor([5.2], dtype=torch.float32)
    expected_prediction = fixed_pred_intercept + torch.matmul(fixed_predictor_values, fixed_pred_coeff)
    predicted_values = glm_pred.predict(fixed_pred_intercept, fixed_pred_coeff)
    
    torch.testing.assert_close(predicted_values, expected_prediction, rtol=1e-4, atol=1e-4)

    # All coefficients and intercept equal to zero should make all predictions zero
    zero_predictions = glm_pred.predict(torch.zeros(1), torch.zeros(fixed_predictor_values.shape[1]))
    torch.testing.assert_close(zero_predictions, torch.zeros(fixed_predictor_values.shape[0]), rtol=1e-4, atol=1e-4)

    # Check sampling
    pyro.set_rng_seed(5)
    preds = glm_pred.sample()

    assert isinstance(preds, dict)
    assert "prediction" in preds.keys()
    assert isinstance(preds["prediction"], torch.Tensor)
    assert preds["prediction"].shape[0] == fixed_predictor_values.shape[0]

    # Check coefficients are included in the results
    assert "coefficients" in preds.keys()

    # Check results agree with manual calculation
    manual_prediction = preds["intercept"] + torch.matmul(fixed_predictor_values, preds["coefficients"])
    torch.testing.assert_close(preds["prediction"], manual_prediction, rtol=1e-4, atol=1e-4)