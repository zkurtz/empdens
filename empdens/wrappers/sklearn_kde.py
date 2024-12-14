import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from ..base import AbstractDensity


def defaults():
    return {}


class SklearnKDE(AbstractDensity):
    def __init__(self, params=None):
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def train(self, data):
        """:param data: (pandas.DataFrame) of numeric features
        """
        assert isinstance(data, pd.DataFrame)
        self.kde = KernelDensity(**self.params)
        _ = self.kde.fit(data.values)

    def density(self, data):
        assert isinstance(data, pd.DataFrame)
        log_dens = self.kde.score_samples(data.values)
        return np.exp(log_dens)
