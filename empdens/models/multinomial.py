"""Model a single categorical feature."""

from typing import Any

import numpy as np
import pandas as pd
from shmistogram.tabulation import SeriesTable, tabulate

from empdens.base import AbstractDensity

DENSITY = "density"
LEVELS = "levels"


class Multinomial(AbstractDensity):
    """Model a single categorical feature."""

    def _train_by_accepting_params(self, counts, values=None):
        self.df = pd.DataFrame({"n_obs": counts})
        if values is not None:
            self.df.index = values

    def train_from_seriestable(self, st: SeriesTable):
        """Train the model from a SeriesTable."""
        assert st.df.shape[0] > 0
        self.name = st.name
        self.df = st.df
        reg_counts = self.df.n_obs.to_numpy() + 1
        self.df[DENSITY] = reg_counts / reg_counts.sum()
        # Assign a tiny prob for never-before observed values of this multinomial:
        self.out_of_sample_dens = 1 / (2 * self.df.n_obs.sum())

    def train(self, df: pd.DataFrame):
        """Specify at least series or counts but not both.

        Args:
            df: A pandas DataFrame with one column of values
        """
        if df.shape[1] > 1:
            raise Exception("Only one-dimensional data is supported")
        series = df[df.columns[0]]
        # if series is not None:
        assert isinstance(series, pd.Series), "Input must be a pandas Series"
        self.name = series.index.name if series.name == "n_obs" else series.name
        assert len(series) > 0, "Empty series"
        st = tabulate(series)
        self.train_from_seriestable(st)

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
            pd.DataFrame({LEVELS: series, "idx": range(len(series))})
            .merge(
                pd.DataFrame({LEVELS: self.df.index.values, DENSITY: self.df[DENSITY].to_numpy()}),
                on=LEVELS,
                how="left",
            )
            .sort_values("idx")
        )
        return df[DENSITY].fillna(self.out_of_sample_dens).to_numpy()

    def rvs(self, n: int = 1) -> pd.DataFrame:
        """Randomly sample from the multinomial distribution."""
        values = np.random.choice(a=self.df.index.to_numpy(), size=n, p=self.df[DENSITY].to_numpy(), replace=True)
        return pd.Series(values, name=self.name).to_frame()
