"""The AbstractDensity class, which is the base class."""

from abc import ABC, abstractmethod

import pandas as pd


class AbstractDensity(ABC):
    """A class of method associated with a density.

    'density' is the only mandatory method, but this includes placeholders for several closely-related methods
    that often would expected in uses cases involving densities.

    Conceptually 'pdf' and 'density' are the same thing, but they differ
    in usage: pdf accepts a single 1-d vector representing a single point,
    while 'density' accepts a pandas DataFrame where each row is a point
    """

    def __init__(self, verbose: bool = False):
        """Initialize the AbstractDensity class."""
        self.verbose = verbose

    def train(self, df: pd.DataFrame) -> None:
        """A method for defining or updating the self.density function base on data."""
        raise Exception("Not yet implemented")

    @abstractmethod
    def density(self, X):
        """Return the density for each row of the pandas DataFrame X as a numpy array."""
        raise Exception("Not yet implemented")

    def predict(self, *args, **kwargs):
        """Prediction alias for density."""
        return self.density(*args, **kwargs)

    def rvs(self, n: int) -> pd.DataFrame:
        """Returns n samples from the space."""
        raise Exception("Not yet implemented")

    def pdf(self, X):
        """Evaluate the density function at a single point."""
        raise Exception("Single-point evaluation not yet supported, but see self.density")

    def vp(self, string):
        """A print function that respects self.verbose."""
        if self.verbose:
            print(string)

    def _validate_data(self, data):
        try:
            assert isinstance(data, pd.DataFrame)
        except Exception:
            raise Exception("the data needs to be a pandas.DataFrame")
        try:
            assert isinstance(data.columns[0], str)
        except Exception:
            raise Exception("the data column names need to be strings, not " + str(type(data.columns[0])))
