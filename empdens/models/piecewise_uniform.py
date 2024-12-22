"""Piecewise uniform density estimator."""

import numpy as np
import pandas as pd
import shmistogram as shmist
from scipy import stats
from shmistogram.binners.bayesblocks import BayesianBlocks

from empdens.base import AbstractDensity
from empdens.models.multinomial import Multinomial


class PiecewiseUniform(AbstractDensity):
    """Adaptive-width histogram density estimator.

    Uses a shmistogram (https://github.com/zkurtz/shmistogram) to separates data
    into
    - 'loner' points (or modes), represented with a multinomial distribution as point masses
    - 'crowd' points, approximated by a standard piecewise uniform distribution
    """

    def __init__(
        self,
        alpha: float | None = None,
        loner_min_count: int = 20,
        binner=None,
        verbose: int = 0,
    ) -> None:
        """Initialize the piecewise uniform density estimator.

        Args:
            alpha: the proportion of loner points
            loner_min_count: minimum number of loner points
            binner: a binner object from shmistogram
            verbose: level of verbosity
        """
        super().__init__()
        self.alpha = alpha
        self.loner_min_count = loner_min_count
        if binner is None:
            self.binner = BayesianBlocks({"sample_size": 10000})
        self.verbose = verbose

    def _uniform(self, bin):
        return stats.uniform(bin.lb, bin.ub - bin.lb)

    def _train_loners(self, loners):
        if loners.n == 0:
            self.multinomial = None
        else:
            m = Multinomial()
            m.train(loners)
            self.multinomial_df = self.loner_crowd_shares[0] * m.df[["density"]]
            self.multinomial = m

    def _set_null_crowd(self):
        self.crowd_uniforms = []
        self.crowd_lookup = pd.DataFrame({"xval": [], "density": []})

    def _train_crowd(self, crowd_bins):
        self.crowd_bins = crowd_bins
        # A multinomial distribution determines which bin to draw from
        if crowd_bins is None:
            self._set_null_crowd()
        elif crowd_bins.shape[0] == 0:
            self._set_null_crowd()
        else:
            self.crowd_multinom = Multinomial()
            self.crowd_multinom.train(df=self.crowd_bins.freq.to_frame())
            self.crowd_uniforms = [self._uniform(row) for _, row in self.crowd_bins.iterrows()]
            # A density lookup for each member of the crowd (assuming asof backward merge)
            crowd_share = self.loner_crowd_shares[1]
            lookup = self.crowd_bins[["lb"]].rename(columns={"lb": "xval"})
            lookup["density"] = crowd_share * self.crowd_bins.rate / self.crowd_bins.freq.sum()
            self.crowd_lookup = pd.concat(
                [lookup, pd.DataFrame({"xval": self.crowd_bins[["ub"]].max().values, "density": [np.nan]})], sort=True
            )

    def train(self, df: pd.DataFrame) -> None:
        """:param series: pandas series of numeric values"""
        assert df.shape[1] == 1, "Only one-dimensional data is supported"
        name = df.columns[0]
        series = df[name]
        shm = shmist.Shmistogram(series, binner=self.binner)
        self.loner_crowd_shares = shm.loner_crowd_shares
        # Loners
        self._train_loners(shm.loners)
        # Crowd
        self._train_crowd(shm.bins)
        # Define density for out-of-sample obs as half the min observed density:
        mmin = np.inf
        if self.multinomial is not None:
            mmin = self.multinomial_df.density.min()
        self.oos_density = min(mmin, self.crowd_lookup.density.min(), 1 / shm.n_obs) / 2

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Compute the density function on each row of X."""
        if X.shape[1] > 1:
            raise ValueError("Only one-dimensional data is supported")
        x_series = X.iloc[:, 0]
        # Identify the unique values for which densities are needed
        ref = pd.DataFrame({"xval": x_series.unique()})
        if self.multinomial is not None:
            # Look up each loner in the multinomial levels; those with no match will be
            #   treated as members of the crowd
            ref = ref.merge(self.multinomial_df, left_on="xval", right_index=True, how="left")
            is_crowd = np.isnan(ref.density)
            ref_loners = ref[~is_crowd].copy()
            ref_crowd = ref[is_crowd].drop("density", axis=1)
        else:
            ref_loners = pd.DataFrame(columns=pd.Index(["xval"]))
            ref_crowd = ref
        ref_crowd = ref_crowd.sort_values(by="xval")  # pyright: ignore
        ref_crowd = ref_crowd.reset_index(drop=True)
        if self.crowd_lookup.xval.dtype == "float64":
            ref_crowd.xval = ref_crowd.xval.astype("float64")
        ref_crowd_roll = pd.merge_asof(ref_crowd, self.crowd_lookup, on="xval")
        assert isinstance(ref_crowd_roll, pd.DataFrame), "ref_crowd_roll is None"
        assert isinstance(ref_loners, pd.DataFrame), "ref_loners is None"
        frames = [ref_loners, ref_crowd_roll]
        nonempty_frames = [frame for frame in frames if not frame.empty]
        final_ref = pd.concat(nonempty_frames, axis=0)
        final_ref["density"] = final_ref.density.fillna(self.oos_density)
        xdf = pd.DataFrame({"xval": x_series.to_numpy(), "order": range(len(X))})
        result = final_ref.merge(xdf, right_on="xval", left_on="xval", how="right").sort_values("order")
        assert result.shape[0] == len(X), "Mismatch in length"
        return result.density.to_numpy()

    def rvs(self, n: int) -> pd.DataFrame:
        """Simulate n draws."""
        # Flip coins to determine number of loners versus crowd
        n_loners = stats.binom.rvs(n=n, p=self.loner_crowd_shares[0], size=1)
        n_crowd = n - n_loners
        # Sample the loners
        if n_loners > 0:
            assert self.multinomial is not None, "Multinomial not defined"
            loners = self.multinomial.rvs(n_loners)
        else:
            loners = np.array([])
        # Sample the crowd
        if n_crowd > 0:
            assert self.crowd_multinom is not None, "Crowd multinomial not defined"
            values_df = self.crowd_multinom.rvs(n_crowd)
            assert values_df.columns.to_list() == ["values"]
            bins = values_df["values"].value_counts()
            crowd_list = [self.crowd_uniforms[k].rvs(bins.iloc[k]) for k in range(len(bins))]
            crowd = np.array([x for y in crowd_list for x in y])
        else:
            crowd = np.array([])
        # Shuffle
        data = np.concatenate((loners, crowd))
        np.random.shuffle(data)
        return pd.DataFrame(data)
