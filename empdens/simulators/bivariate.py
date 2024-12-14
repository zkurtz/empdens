import numpy as np
import pandas as pd
from scipy import stats

from empdens.base import AbstractDensity


class Zena(AbstractDensity):
    """Zena (arbitrary name) -- a bivariate data simulator."""

    def __init__(self):
        super().__init__()
        self.gauss = stats.truncnorm(-2, 4)
        self.triang = stats.triang(0, 0, 3)

    def rvs(self, n):
        """Simulate a simple bivariate density.

        The density is 2-dimensional with a discontinuous covariance structure
        """
        return pd.DataFrame({"gaussian": self.gauss.rvs(size=n), "triangular": self.triang.rvs(size=n)})

    def density(self, points):
        if isinstance(points, pd.DataFrame):
            points = points.values
        return np.array([self.gauss.pdf(p[0]) * self.triang.pdf(p[1]) for p in points])
