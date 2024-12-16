"""Classifier-adjusted density estimation."""

import numpy as np
import pandas as pd
from sklearn import metrics

from empdens import models
from empdens.base import AbstractDensity
from empdens.classifiers.base import AbstractLearner
from empdens.classifiers.lightgbm import Lgbm
from empdens.data import CadeData


def auc(df):
    """Compute the area under the ROC curve."""
    fpr, tpr, _ = metrics.roc_curve(df.truth.values, df.pred.values, pos_label=1)
    return metrics.auc(fpr, tpr)


class Cade(AbstractDensity):
    """Classifier-adjusted density estimation.

    Based on https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf
    """

    # A soft target for the number of instances to simulate when `sim_size` is "auto"
    simulation_size_attractor = 10000

    def __init__(
        self,
        initial_density: AbstractDensity | None = None,
        classifier: AbstractLearner | None = None,
        sim_size: str = "auto",
        verbose: bool = False,
    ):
        """Initialize the classifier-adjusted density estimation model.

        Args:
            initial_density: A naive density model to use as the basis for CADE estimation. CADE uses this as a
                reference such that the final estimated density is a product of the initial density and the classifier
                adjustment. If None, defaults to a JointDensity model.
            classifier: A classifier to use for the adjustment. If None, defaults to a LightGBM classifier.
            sim_size: The number of synthetic samples to simulate. If "auto", the simulation size is set as the
                geometric mean between the data size and `simulation_size_attractor`. If a positive number less than
                100, the simulation size is `round(sim_size)*df.shape[0]`. If `sim_size` is greater than or equal to
                100, the simulation size is `round(sim_size)`.
            verbose: Whether to print diagnostic information.
        """
        super().__init__()
        self.initial_density = initial_density or models.JointDensity()
        self.classifier = classifier or Lgbm()
        self.sim_size = sim_size
        self.verbose = verbose

    def compute_simulation_size(self, df):
        """Determine the number of synthetic data samples to simulate.

        If self.sim_size is 'auto', sets the simulation size as the geometric mean
        between the data size and self.simulation_size_attractor

        If self.sim_size is a positive number less than 100, simulation size is
        round(self.sim_size)*df.shape[0]

        Finally, if self.sim_size >= 100, simulation size is round(self.sim_size)
        """
        n_real = df.shape[0]
        if isinstance(self.sim_size, str):
            assert self.sim_size == "auto"
            sim_n = np.sqrt(n_real * self.simulation_size_attractor)
        elif self.sim_size < 100:
            assert self.sim_size > 0
            sim_n = round(self.sim_size * n_real)
            if sim_n < 10:
                raise Exception("Simulation size is very small. Consider using a larger value of sim_size")
        else:
            sim_n = round(self.sim_size)
        self.sim_rate = sim_n / df.shape[0]
        return int(sim_n)

    def _diagnostics(self, x, truth):
        val_df = pd.DataFrame({"pred": self.classifier.predict(x), "truth": truth})
        self.diagnostics = {
            "val_df": val_df,
            "auc": auc(val_df),
        }

    def _validate_data(self, data):
        try:
            assert isinstance(data, pd.DataFrame)
        except Exception:
            raise Exception("the data needs to be a pandas.DataFrame")
        try:
            assert isinstance(data.columns[0], str)
        except Exception:
            raise Exception("the data column names need to be strings, not " + str(type(data.columns[0])))

    def train(self, df, diagnostics=False):
        """Model the density of the data.

        :param df: (pandas DataFrame)
        """
        self._validate_data(df)
        self.vp("Training a generative density model on " + str(df.shape[0]) + " samples")
        self.initial_density.train(df)
        sim_n = self.compute_simulation_size(df)
        self.vp("Simulating " + str(sim_n) + " fake samples from the model and join it with the real data")
        xdf = pd.concat([df, self.initial_density.rvs(sim_n)])
        assert isinstance(xdf, pd.DataFrame), "CadeData.X requires a pandas DataFrame"
        partially_synthetic_data = CadeData(
            X=xdf,
            y=np.concatenate([np.ones(df.shape[0]), np.zeros(sim_n)]),
        )
        self.vp("Train the classifier to distinguish real from fake")
        self.classifier.train(partially_synthetic_data)
        if diagnostics:
            self._diagnostics(partially_synthetic_data.X, partially_synthetic_data.y)
        if self.verbose:
            if not hasattr(self, "diagnostics"):
                self._diagnostics(partially_synthetic_data.X, partially_synthetic_data.y)
            AUROC = str(round(self.diagnostics["auc"], 3))
            print("In-sample, the classifier had AUROC = " + AUROC)

    def density(self, X):
        """Predict the density at new points.

        Apply equation 2.1 in https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf

        :param X: (pd.DataFrame or numpy array) Must match the exact column order of the `df`
            argument that was passed to self.train
        """
        self._validate_data(X)
        # Initial density estimate
        synthetic_dens = self.initial_density.density(X)
        # Classifier adjustment factor
        p_real = self.classifier.predict(X)
        odds_real = p_real / (1 - p_real)
        classifier_adjustment = self.sim_rate * odds_real
        return synthetic_dens * classifier_adjustment
