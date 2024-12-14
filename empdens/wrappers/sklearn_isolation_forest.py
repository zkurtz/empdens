import pandas as pd
from sklearn.ensemble import IsolationForest

from empdens.base import AbstractDensity


def defaults():
    return {"behaviour": "new", "contamination": "auto"}


class SklearnIsolationForest(AbstractDensity):
    def __init__(self, params=None):
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def train(self, data):
        """:param data: (pandas.DataFrame) of numeric features"""
        assert isinstance(data, pd.DataFrame)
        self.forest = IsolationForest(**self.params)
        _ = self.forest.fit(data.values)

    def density(self, data):
        assert isinstance(data, pd.DataFrame)
        # TODO: explain how this is kinda sorta getting a density
        dens = 1 + self.forest.score_samples(data.values)
        return dens
