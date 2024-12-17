"""Wrapper for sklearn's KernelDensity class."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from empdens.base import AbstractDensity


def defaults():
    """Default parameters for the KDE."""
    return {}


class SklearnKDE(AbstractDensity):
    """Wrapper for sklearn's KernelDensity class."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the KDE model.

        Args:
            params: Additional named arguments to be passed to `sklearn.neighbors.KernelDensity`.
        """
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def train(self, df: pd.DataFrame) -> None:
        """Train the KDE model.

        Args:
            df: Data to train the model on.
        """
        self.kde = KernelDensity(**self.params)
        _ = self.kde.fit(df.to_numpy())

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate the density of the data at the given points."""
        log_dens = self.kde.score_samples(X.to_numpy())
        return np.exp(log_dens)
