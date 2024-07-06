import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pandas as pd

class AbstractRegressionPrediction(PyroModule, metaclass=ABCMeta):
    @abstractmethod
    def predict(self):
        """
        Make a regression prediction.
        """
        pass

    @abstractmethod
    def sample(self, obs=None):
        """
        Observe or sample from the regression problem according to the specified distributions.
        """
        pass

class GLMPrediction(AbstractRegressionPrediction):
    def __init__(self, name: str, fixed_predictor_values: torch.Tensor, 
                 intercept_prior: dist.Distribution, coefficient_priors: dist.Distribution,
                 transform=None, intercept_suffix="_intercept", coefficient_suffix="_coefficients"):
        super().__init__()
        if transform is None:
            transform = lambda x: x  # Identity transform

        self.name = name
        self.fixed_predictor_values = fixed_predictor_values
        self.transform = transform
        self.intercept_prior = intercept_prior
        self.coefficient_priors = coefficient_priors
        self.intercept_suffix = intercept_suffix
        self.coefficient_suffix = coefficient_suffix

    def predict(self, intercept: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Generates a transformed prediction with intercept, coefficients, and fixed predictor values.
        """
        transformed_prediction = intercept + torch.matmul(self.fixed_predictor_values, coefficients)
        return self.transform(transformed_prediction)

    def sample(self, obs=None):
        """
        Sample from the generalized linear model.
        """
        intercept = pyro.sample(self.name + self.intercept_suffix, self.intercept_prior)
        coefficients = pyro.sample(self.name + self.coefficient_suffix, self.coefficient_priors)
        prediction = self.predict(intercept, coefficients)
        return {'prediction': prediction, 'intercept': intercept, 'coefficients': coefficients}

    def __repr__(self):
        return f"GLMPrediction({self.name})"