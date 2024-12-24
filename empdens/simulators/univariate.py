"""Simulators for univariate densities."""

import numpy as np
import pandas as pd
from scipy import stats

from empdens.base import AbstractDensity


class Multinomial:
    """A multinomial random variable."""

    def __init__(self, probs: np.ndarray) -> None:
        """Define a multinomial random variable object.

        Args:
            probs: The probability of each class, with classes indexed as 0 to len(probs)-1
        """
        assert isinstance(probs, np.ndarray)
        self.idx = list(range(len(probs)))
        self.probs = probs / probs.sum()

    def rvs(self, n: int) -> np.ndarray:
        """Simulate n draws from the multinomial distribution."""
        return np.random.choice(a=self.idx, size=n, p=self.probs, replace=True)

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate the density at the given points."""
        assert X.shape[1] == 1
        points = X.to_numpy()[:, 0]
        return np.array([self.probs[k] for k in points])


class BartSimpson(AbstractDensity):
    """The "claw".

    As in https://projecteuclid.org/download/pdf_1/euclid.aos/1176348653;
    renamed as in http://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf.
    """

    def __init__(self) -> None:
        """Initialize the Bart Simpson density."""
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

    def rvs(self, n: int) -> pd.DataFrame:
        """Simulate n draws."""
        idxs = self.multinomial.rvs(n)
        values, counts = np.unique(idxs, return_counts=True)
        samples = [self.gaussians[values[k]].rvs(counts[k]) for k in range(len(values))]
        samples = [v for sublist in samples for v in sublist]
        np.random.shuffle(samples)
        return pd.DataFrame({"bart_simpson": samples})

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate the density at the given points."""
        assert X.shape[1] == 1
        points = X.to_numpy()[:, 0]
        pdf_funcs = [gauss.pdf for gauss in self.gaussians]  # pyright: ignore
        return np.column_stack([func(points) for func in pdf_funcs]) @ self.multinomial.probs

    def plot(self, n: int = 200, xlims: list[float | int] = [-2, 2]) -> None:
        """Plot the density."""
        dfg = pd.DataFrame({"x": np.linspace(xlims[0], xlims[1], n)})
        dfg["generative density"] = self.density(dfg)
        ax = dfg.plot(x="x", y="generative density")
        ax.get_legend().remove()
