import numpy as np
import pandas as pd
from scipy import stats

from ..base import AbstractDensity


class Multinomial:
    def __init__(self, probs):
        """Define a multinomial random variable object

        :param probs: The probability of each class, with classes indexed as 0 to len(probs)-1
        """
        assert isinstance(probs, np.ndarray)
        self.idx = list(range(len(probs)))
        self.probs = probs / probs.sum()

    def rvs(self, n):
        return np.random.choice(a=self.idx, size=n, p=self.probs, replace=True)

    def density(self, points):
        if isinstance(points, pd.DataFrame):
            assert points.shape[1] == 1
            points = points.values[:, 0]
        return [self.probs[k] for k in points]


class BartSimpson(AbstractDensity):
    """The "claw" in https://projecteuclid.org/download/pdf_1/euclid.aos/1176348653;
    renamed as in http://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf
    """

    def __init__(self):
        super().__init__()
        # Guassians to mix over
        self.gaussians = [
            stats.norm(),
            stats.norm(loc=-1, scale=0.1),
            stats.norm(loc=-0.5, scale=0.1),
            stats.norm(loc=0, scale=0.1),
            stats.norm(loc=0.5, scale=0.1),
            stats.norm(loc=1, scale=0.1),
        ]
        # Mixing weights
        self.multinomial = Multinomial(probs=np.array([0.5] + [0.1] * 5))

    def rvs(self, n):
        """Simulate n draws"""
        idxs = self.multinomial.rvs(n)
        values, counts = np.unique(idxs, return_counts=True)
        samples = [self.gaussians[values[k]].rvs(counts[k]) for k in range(len(values))]
        samples = [v for sublist in samples for v in sublist]
        np.random.shuffle(samples)
        return pd.DataFrame({"bart_simpson": samples})

    def density(self, points):
        if isinstance(points, pd.DataFrame):
            assert points.shape[1] == 1
            points = points.values[:, 0]
        return np.column_stack([g.pdf(points) for g in self.gaussians]) @ self.multinomial.probs

    def plot(self, n=200, xlims=[-2, 2]):
        dfg = pd.DataFrame({"x": np.linspace(xlims[0], xlims[1], n)})
        dfg["generative density"] = self.density(dfg.x.values)
        ax = dfg.plot(x="x", y="generative density")
        ax.get_legend().remove()
