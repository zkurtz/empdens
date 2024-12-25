"""Bivariate data simulators."""

import numpy as np
import pandas as pd
from scipy import stats

from empdens.base import AbstractDensity


class Zena(AbstractDensity):
    """A simple bivariate data simulator.

    This use two independent distributions, a Gaussian and a triangular distribution.
    """

    def __init__(self):
        """Initialize the bivariate data simulator."""
        super().__init__()
        self.gauss = stats.truncnorm(-2, 4)
        self.triang = stats.triang(0, 0, 3)

    def rvs(self, n: int) -> pd.DataFrame:
        """Simulate draws from the joint distribution."""
        return pd.DataFrame({"gaussian": self.gauss.rvs(size=n), "triangular": self.triang.rvs(size=n)})

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate the density at the given points."""
        points = X.to_numpy()
        gauss_pdf = self.gauss.pdf  # pyright: ignore
        triang_pdf = self.triang.pdf  # pyright: ignore
        return np.array([gauss_pdf(p[0]) * triang_pdf(p[1]) for p in points])
