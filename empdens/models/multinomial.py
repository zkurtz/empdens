import pdb

import numpy as np
import pandas as pd
from shmistogram.tabulation import SeriesTable

from empdens.base import AbstractDensity


class Multinomial(AbstractDensity):
    """Model a single categorical feature."""

    def _density(self):
        reg_counts = self.df.n_obs.values + 1
        self.df["density"] = reg_counts / reg_counts.sum()
        # Assign a tiny prob for never-before observed values of this multinomial:
        self.out_of_sample_dens = 1 / (2 * self.df.n_obs.sum())

    def _train_empirically(self, series):
        if isinstance(series, SeriesTable):
            st = series
            assert series.df.shape[0] > 0
        elif isinstance(series, pd.Series):
            assert len(series) > 0
            st = SeriesTable(series)
        self.name = st.name
        self.df = st.df

    def _train_by_accepting_params(self, counts, values=None):
        self.df = pd.DataFrame({"n_obs": counts})
        if values is not None:
            pdb.set_trace()
            self.df.index = values

    def train(self, series=None, counts=None, values=None):
        """Specify at least series or counts but not both
        :param series: (pandas.Series or SeriesTable of integers
        :param counts: numpy 1-d array of counts corresponding to
        each entry of `values`
        :param values: numpy 1-d array; integers representing each multinomial outcome;
        ignored if `counts` is None.
        """
        if series is not None:
            self._train_empirically(series)
        else:
            assert counts is not None
            assert counts.sum() > 0
            self._train_by_accepting_params(counts, values=values)
        self._density()

    def density(self, x):
        """Compute the density for an individual value."""
        try:
            return self.df.density[x]
        except:
            # assert x not in self.df.index.values
            return self.out_of_sample_dens

    def density_series(self, x):
        """Fast density computation for a list of values.

        :param x: (pandas.Series) Values at which to compute the density
        """
        assert isinstance(x, pd.Series)
        df = (
            pd.DataFrame({"levels": x, "idx": range(len(x))})
            .merge(
                pd.DataFrame({"levels": self.df.index.values, "density": self.df.density.values}),
                on="levels",
                how="left",
            )
            .sort_values("idx")
        )
        return df.density.fillna(self.out_of_sample_dens).values

    def rvs(self, n=1):
        return np.random.choice(a=self.df.index.values, size=n, p=self.df.density.values, replace=True)
