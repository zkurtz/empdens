import pdb

import numpy as np
import pandas as pd
import shmistogram as shmist
from scipy import stats

from ..base import AbstractDensity
from .multinomial import Multinomial


class PiecewiseUniform(AbstractDensity):
    """Adaptive-width histogram density estimator

    Uses a shmistogram (https://github.com/zkurtz/shmistogram) to separates data
    into
    - 'loner' points (or modes), represented with a multinomial distribution as point masses
    - 'crowd' points, approximated by a standard piecewise uniform distribution
    """

    def __init__(self, alpha=None, loner_min_count=20, binner=None, verbose=0):
        super().__init__()
        self.alpha = alpha
        self.loner_min_count = loner_min_count
        if binner is None:
            self.binner = shmist.binners.BayesianBlocks({"sample_size": 10000})
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
            self.crowd_multinom.train(counts=self.crowd_bins.freq.values)
            self.crowd_uniforms = [self._uniform(row) for _, row in self.crowd_bins.iterrows()]
            # A density lookup for each member of the crowd (assuming asof backward merge)
            crowd_share = self.loner_crowd_shares[1]
            lookup = self.crowd_bins[["lb"]].rename(columns={"lb": "xval"})
            lookup["density"] = crowd_share * self.crowd_bins.rate / self.crowd_bins.freq.sum()
            self.crowd_lookup = pd.concat(
                [lookup, pd.DataFrame({"xval": self.crowd_bins[["ub"]].max().values, "density": [np.nan]})], sort=True
            )

    def train(self, series):
        """:param series: pandas series of numeric values

        """
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

    def density(self, x):
        """Compute the density function on each row of x
        """
        # Identify the unique values for which densities are needed
        ref = pd.DataFrame({"xval": x.unique()})
        if self.multinomial is not None:
            # Look up each loner in the multinomial levels; those with no match will be
            #   treated as members of the crowd
            ref = ref.merge(self.multinomial_df, left_on="xval", right_index=True, how="left")
            is_crowd = np.isnan(ref.density)
            ref_loners = ref[~is_crowd].copy()
            ref_crowd = ref[is_crowd].drop("density", axis=1)
        else:
            ref_loners = None
            ref_crowd = ref
        ref_crowd = ref_crowd.sort_values("xval").reset_index(drop=True)
        if self.crowd_lookup.xval.dtype == "float64":
            ref_crowd.xval = ref_crowd.xval.astype("float64")
        ref_crowd_roll = pd.merge_asof(ref_crowd, self.crowd_lookup, on="xval")
        final_ref = pd.concat([ref_loners, ref_crowd_roll])
        final_ref["density"] = final_ref.density.fillna(self.oos_density)
        xdf = pd.DataFrame({"xval": x.values, "order": range(len(x))})
        result = final_ref.merge(xdf, right_on="xval", left_on="xval", how="right").sort_values("order")
        assert result.shape[0] == len(x)
        return result.density.values

    def rvs(self, n):
        # Flip coins to determine number of loners versus crowd
        n_loners = stats.binom.rvs(n=n, p=self.loner_crowd_shares[0], size=1)[0]
        n_crowd = n - n_loners
        # Sample the loners
        if n_loners > 0:
            loners = self.multinomial.rvs(n_loners)
        else:
            loners = np.array([])
        # Sample the crowd
        if n_crowd > 0:
            try:
                assert self.crowd_multinom is not None
            except:
                pdb.set_trace()
            bins = pd.Series(self.crowd_multinom.rvs(n_crowd)).value_counts()
            crowd_list = [self.crowd_uniforms[k].rvs(bins[k]) for k in bins.index.values]
            crowd = np.array([x for y in crowd_list for x in y])
        else:
            crowd = np.array([])
        # Shuffle
        data = np.concatenate((loners, crowd))
        np.random.shuffle(data)
        return data
