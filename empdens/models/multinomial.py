"""Model a single categorical feature."""

from typing import Any

import numpy as np
import pandas as pd
from pandahandler.tabulation import Tabulation, tabulate

from empdens.base import AbstractDensity

DENSITY = "density"
LEVELS = "levels"


class Multinomial(AbstractDensity):
    """Model a single categorical feature."""

    def train_from_tabulation(self, counts: Tabulation) -> None:
        """Train the model from a tabulation.

        Args:
            counts: A tabulation of the distinct values.
        """
        self.counts = counts
        if self.counts.n_distinct < 1:
            raise ValueError("No distinct values in the tabulation")
        _inflated_counts = self.counts.counts + 1
        self.regularized_rates = _inflated_counts / _inflated_counts.sum()
        # Assign a tiny prob for never-before observed values of this multinomial:
        self.out_of_sample_rate = 1 / (2 * self.counts.n_values)

    def train(self, df: pd.DataFrame):
        """Specify at least series or counts but not both.

        Args:
            df: A pandas DataFrame with one column of values
        """
        if df.shape[1] > 1:
            raise Exception("Only one-dimensional data is supported")
        series = df[df.columns[0]]
        assert isinstance(series, pd.Series), "Expected a series"
        counts = tabulate(series)
        self.train_from_tabulation(counts=counts)

    def point_density(self, item: Any) -> float:
        """Compute the density for an individual value."""
        if item in self.regularized_rates.index:
            return self.regularized_rates.loc[item]
        return self.out_of_sample_rate

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Fast density computation for a list of values.

        Args:
            X: A data frame with one column of values.
        """
        if X.shape[1] > 1:
            raise Exception("Only one-dimensional data is supported")
        series = X[X.columns[0]]
        assert isinstance(series, pd.Series)
        base = pd.DataFrame({LEVELS: series, "idx": range(len(series))})
        other = pd.DataFrame(
            {
                LEVELS: self.counts.counts.index.to_numpy(),
                DENSITY: self.regularized_rates.to_numpy(),
            }
        )
        df = base.merge(other, on=LEVELS, how="left").sort_values("idx")
        return df[DENSITY].fillna(self.out_of_sample_rate).to_numpy()

    def rvs(self, n: int = 1) -> pd.DataFrame:
        """Randomly sample from the multinomial distribution."""
        values = np.random.choice(
            a=self.counts.counts.index.to_numpy(),
            size=n,
            p=self.regularized_rates.to_numpy(),
            replace=True,
        )
        return pd.Series(values, name=self.counts.name).to_frame()
