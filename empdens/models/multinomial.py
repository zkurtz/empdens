"""Model a single categorical feature."""

from typing import Any

import numpy as np
import pandas as pd
from shmistogram.tabulation import SeriesTable, tabulate

from empdens.base import AbstractDensity


class Multinomial(AbstractDensity):
    """Model a single categorical feature."""

    def _density(self):
        reg_counts = self.df.n_obs.to_numpy() + 1
        self.df["density"] = reg_counts / reg_counts.sum()
        # Assign a tiny prob for never-before observed values of this multinomial:
        self.out_of_sample_dens = 1 / (2 * self.df.n_obs.sum())

    def _train_empirically(self, series: SeriesTable | pd.Series):
        if isinstance(series, SeriesTable):
            st = series
            assert st.df.shape[0] > 0
            self.name = st.name
        else:
            assert isinstance(series, pd.Series), f"Input must be a pandas Series, got {type(series)}"
            self.name = series.index.name if series.name == "n_obs" else series.name
            assert len(series) > 0, "Empty series"
            st = tabulate(series)
        self.df = st.df

    def _train_by_accepting_params(self, counts, values=None):
        self.df = pd.DataFrame({"n_obs": counts})
        if values is not None:
            self.df.index = values

    def train(self, df: pd.DataFrame, counts=None, values=None):
        """Specify at least series or counts but not both.

        :param series: (pandas.Series or SeriesTable of integers
        :param counts: numpy 1-d array of counts corresponding to
        each entry of `values`
        :param values: numpy 1-d array; integers representing each multinomial outcome;
        ignored if `counts` is None.
        """
        if df.shape[1] > 1:
            raise Exception("Only one-dimensional data is supported")
        series = df[df.columns[0]]
        # if series is not None:
        assert isinstance(series, pd.Series), "Input must be a pandas Series"
        self._train_empirically(series)
        # else:
        #     assert counts is not None
        #     assert counts.sum() > 0
        #     self._train_by_accepting_params(counts, values=values)
        self._density()

    def point_density(self, item: Any) -> np.ndarray:
        """Compute the density for an individual value. TODO: should not be for individual value."""
        raise NotImplementedError

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Fast density computation for a list of values.

        Args:
            X: A data frame with one column of values.
        """
        if X.shape[1] > 1:
            raise Exception("Only one-dimensional data is supported")
        series = X[X.columns[0]]
        assert isinstance(series, pd.Series)
        df = (
            pd.DataFrame({"levels": series, "idx": range(len(series))})
            .merge(
                pd.DataFrame({"levels": self.df.index.values, "density": self.df.density.values}),
                on="levels",
                how="left",
            )
            .sort_values("idx")
        )
        return df.density.fillna(self.out_of_sample_dens).to_numpy()

    def rvs(self, n: int = 1) -> pd.DataFrame:
        """Randomly sample from the multinomial distribution."""
        values = np.random.choice(a=self.df.index.to_numpy(), size=n, p=self.df.density.to_numpy(), replace=True)
        return pd.Series(values, name=self.name).to_frame()
