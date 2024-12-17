"""Wrapper for fastKDE package."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate

from empdens.base import AbstractDensity


def defaults():
    """Default parameters for the fastKDE."""
    return {}


class Interpolator:
    """Interpolates a grid of values in N-space to estimate values at arbitrary points."""

    def __init__(
        self,
        grid: np.ndarray,
        axes: list[np.ndarray],
        params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the interpolator with a grid of values and the axes of the grid.

        Args:
            grid: This is an n-dimensional function surface.
            axes: Position labels for each dimension of `grid`.
            params: Additional named arguments to be passed to `scipy.interpolate.griddata` to control how the
                interpolation is done.
        """
        self.params = {}
        if params is not None:
            assert isinstance(params, dict)
            self.params = params
        self.values = grid.ravel()
        idx = np.meshgrid(*[np.arange(len(ax)) for ax in axes])
        indices = [ix.ravel() for ix in idx]
        self.positions = np.vstack([a[i] for a, i in zip(axes, indices)]).T

    def interpolate(self, x: np.ndarray) -> np.ndarray:
        """Interpolate the values at arbitrary points in N-space.

        Args:
            x: Points in the original N-space at which to estimate values.

        Returns:
            Interpolated values at the positions `x` based on the `self.values` observed over the `self.grid`.
        """
        return interpolate.griddata(self.positions, self.values, x, **self.params)


class FastKDE(AbstractDensity):
    """Wrapper for fastKDE package."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize the fastKDE model.

        Args:
            params: Additional named arguments to be passed to `fastkde.fastKDE
        """
        self._load_fastKDE()
        super().__init__()
        self.params = defaults()
        if params is not None:
            self.params.update(params)

    def _load_fastKDE(self) -> None:
        """Load the fastKDE package."""
        try:
            from fastkde import fastKDE
        except Exception as err:
            raise Exception("""
                You need to install fastKDE first.
                We recommend installing from source due to
                https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for

                To install fastKDE from source, do
                (1) `pip install cython`
                (2) `pip install numpy`
                (3) `pip install git+https://bitbucket.org/lbl-cascade/fastkde.git#egg=fastkde`
            """) from err
        self.fastKDE = fastKDE

    def train(self, df: pd.DataFrame) -> None:
        """Train the density estimator on the given data.

        Currently throws a warning:
            https://bitbucket.org/lbl-cascade/fastkde/issues/5/using-a-non-tuple-sequence-for

        Args:
            df: Numeric features.
        """
        with warnings.catch_warnings():
            msg = "Using a non-tuple sequence for multidimensional indexing is deprecated"
            warnings.filterwarnings("ignore", message=msg)
            fkde = self.fastKDE.pdf(*[df[col].to_numpy() for col in df], **self.params)
            grid = fkde.data
            axes = [getattr(fkde, dim_name).to_numpy() for dim_name in fkde.dims]
        self.interpolator = Interpolator(grid, axes)

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate the density at the given points."""
        return self.interpolator.interpolate(X.to_numpy())
