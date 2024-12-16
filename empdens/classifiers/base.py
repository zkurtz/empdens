"""Base class for all classifiers in the empdens package."""

import copy
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractLearner(ABC):
    """Base class for all classifiers in the empdens package."""

    def __init__(self, params: dict[str, Any] | None = None, verbose=False):
        """Initialize the base class for all classifiers in the empdens package."""
        self.params = self.default_params()
        if params is not None:
            self.params.update(copy.deepcopy(params))
        self.verbose = verbose

    @abstractmethod
    def default_params(self) -> dict[str, Any]:
        """Return default parameters for the classifier."""
        pass

    @abstractmethod
    def train(self, data):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Predict on new data."""
        pass

    def vp(self, string):
        """Prints a string if verbose is True."""
        if self.verbose:
            print(string)
