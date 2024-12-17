"""Wrapper for sklearn's IsolationForest."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from empdens.base import AbstractDensity


def defaults():
    """Default parameters for the IsolationForest."""
    return {"contamination": "auto"}


class SklearnIsolationForest(AbstractDensity):
    """Wrapper for sklearn's IsolationForest."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the IsolationForest model.

        Args:
            params: Additional named arguments to be passed to `sklearn.ensemble.IsolationForest`.
        """
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def train(self, df: pd.DataFrame) -> None:
        """Train the model.

        Args:
            df: Data to train the model on.
        """
        self.forest = IsolationForest(**self.params)
        _ = self.forest.fit(df.to_numpy())

    def density(self, df: pd.DataFrame) -> np.ndarray:
        """Estimate the density of the data at the given points.

        Args:
            df: Data to estimate the density over.
        """
        # TODO: explain how this is kinda sorta getting a density
        dens = 1 + self.forest.score_samples(df.to_numpy())
        return dens
