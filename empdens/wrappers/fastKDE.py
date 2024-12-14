import warnings

import numpy as np
import pandas as pd
from scipy import interpolate

from empdens.base import AbstractDensity


def defaults():
    return {}


class Interpolator(object):
    def __init__(self, grid, axes, params=None):
        """:param grid: (np.ndarray of floats) This is an n-dimensional function surface
        :param axes: (list of 1-D numpy arrays) Position labels for each dimension of `grid`
        :param params: (dict) additional named arguments to be passed to
            `scipy.interpolate.griddata` to control how the interpolation is done
        """
        self.params = {}
        if params is not None:
            assert isinstance(params, dict)
            self.params = params
        self.values = grid.ravel()
        idx = np.meshgrid(*[np.arange(len(ax)) for ax in axes])
        indices = [ix.ravel() for ix in idx]
        self.positions = np.vstack([a[i] for a, i in zip(axes, indices)]).T

    def interpolate(self, x):
        """:param x: (2-D numpy array) with number of columns equal to len(self.axes);
            these are points in the original N-space at which to estimate values
        :return: Interpolated values at the positions `x` based on the `self.values`
        observed over the `self.grid`
        """
        return interpolate.griddata(self.positions, self.values, x, **self.params)


class FastKDE(AbstractDensity):
    def __init__(self, params=None):
        self._load_fastKDE()
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def _load_fastKDE(self):
        try:
            from fastkde import fastKDE
        except:
            raise Exception("""
                You need to install fastKDE first.
                We recommend installing from source due to
                https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for

                To install fastKDE from source, do
                (1) `pip install cython`
                (2) `pip install numpy`
                (3) `pip install git+https://bitbucket.org/lbl-cascade/fastkde.git#egg=fastkde`
            """)
        self.fastKDE = fastKDE

    def train(self, data):
        """Currently throws a warning:
            https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for
        :param data: (pandas.DataFrame) of numeric features.
        """
        assert isinstance(data, pd.DataFrame)
        with warnings.catch_warnings():
            msg = "Using a non-tuple sequence for multidimensional indexing is deprecated"
            warnings.filterwarnings("ignore", message=msg)
            grid, axes = self.fastKDE.pdf(*[data[col].values for col in data.columns], **self.params)
        if not isinstance(axes, list):  # then probably this is 1-D estimation
            axes = [axes]
        self.interpolator = Interpolator(grid, axes)

    def density(self, data):
        assert isinstance(data, pd.DataFrame)
        return self.interpolator.interpolate(data.values)
