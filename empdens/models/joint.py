"""Model a joint density of multiple features."""

from typing import Any

import numpy as np
import pandas as pd

from empdens.base import AbstractDensity
from empdens.models.multinomial import Multinomial
from empdens.models.piecewise_uniform import PiecewiseUniform


class JointDensity(AbstractDensity):
    """Model a joint density of multiple features."""

    def __init__(self, numeric_params: dict[str, Any] | None = None, verbose: bool = False) -> None:
        """Initialize the joint density model."""
        super().__init__(verbose=verbose)
        self.multinom_class = Multinomial
        self.numeric_class = PiecewiseUniform
        self.numeric_params = numeric_params

    def _fit_categorical(self, series: pd.Series) -> Multinomial:
        model = self.multinom_class()
        model.train(series.to_frame())
        return model

    def _fit_continuous(self, values: pd.Series) -> PiecewiseUniform:
        params = {}
        if self.numeric_params is not None:
            params = self.numeric_params
        model = self.numeric_class(verbose=self.verbose - 1, **params)
        model.train(values.to_frame())
        return model

    def _fit_univarite(self, series: pd.Series) -> Multinomial | PiecewiseUniform:
        msg = "Fitting univariate density on " + str(series.name) + " as "
        if isinstance(series.dtype, pd.CategoricalDtype):
            self.vp(msg + "categorical")
            return self._fit_categorical(series)
        else:
            self.vp(msg + "continuous")
            return self._fit_continuous(series)

    def train(self, df: pd.DataFrame) -> None:
        """Train the joint density model on a DataFrame."""
        self.columns = df.columns
        self.univariates = {v: self._fit_univarite(df[v]) for v in self.columns}  # pyright: ignore

    def density(self, X: pd.DataFrame, log: bool = False):
        """Compute the density for an individual value."""
        assert isinstance(X, pd.DataFrame)
        assert all(X.columns == self.columns)
        df_log_univariate = pd.DataFrame(
            {v: np.log(self.univariates[v].density(X[[v]])) for v in self.columns},  # pyright: ignore
        )
        log_dens = df_log_univariate.sum(axis=1).to_numpy()
        if log:
            return log_dens
        return np.exp(log_dens)

    def rvs(self, n):
        """Generate n samples from the fitted distribution."""
        if not hasattr(self, "univariates"):
            raise Exception("Call `train` before you call `rvs`")
        samples = [self.univariates[col].rvs(n) for col in self.columns]
        return pd.concat(samples, axis=1)
