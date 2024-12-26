"""Classifier-adjusted density estimation."""

import numpy as np
import pandas as pd
from pandahandler import Schema, categorize_non_numerics
from sklearn import metrics

from empdens import models
from empdens.base import AbstractDensity
from empdens.classifiers.base import AbstractLearner
from empdens.classifiers.lightgbm import Lgbm
from empdens.data import CadeData

AUROC = "auroc"
PRED = "pred"
TRUTH = "truth"


def auroc(df):
    """Compute the area under the ROC curve.

    Args:
        df: Data frame containing columns `truth` and `pred`.
    """
    fpr, tpr, _ = metrics.roc_curve(df[TRUTH].to_numpy(), df[PRED].to_numpy(), pos_label=1)
    return metrics.auc(fpr, tpr)


def compute_simulation_size(n_real: int) -> int:
    """Determine the number of synthetic data samples to simulate.

    Typically, the larger the simulation size, the more accurate the density estimate. However, we also want to
    be concious of the computational cost. Consider two extremes:
     - Data are extremely small, <100 rows. Then we can easily simulate 100x the data size.
     - Data are large, >100k rows. We can typically afford to match the data size but not much more than that.
    To achieve a smooth continuum between these two extremes, we simulate (1.0 + extra) times the data size, where
    `extra > 0` and decreases as the data size increases. Specifically, we define
    `extra = 100 / sqrt(data size)`. Examples:
        - 100 rows -> extra = 100 / 10 = 10 -> 1100 simulated rows
        - 10000 rows -> extra = 100 / 100 = 1 -> 20000 simulated rows
        - 1000000 rows -> extra = 100 / 1000 = 0.1 -> 1001000 simulated rows

    Args:
        n_real: Number of real data samples.
    """
    extra = 100 / np.sqrt(n_real)
    approx_size = (1.0 + extra) * n_real
    return round(approx_size)


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
        sim_size: int | None = None,
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
        self.initial_density = initial_density or models.JointDensity(verbose=verbose)
        self.classifier = classifier or Lgbm()
        self.sim_size = sim_size
        self.verbose = verbose

    def _diagnostics(self, x, truth):
        val_df = pd.DataFrame({PRED: self.classifier.predict(x), TRUTH: truth})
        self.diagnostics = {
            "val_df": val_df,
            AUROC: auroc(val_df),
        }

    def train(self, df: pd.DataFrame, diagnostics: bool = False):
        """Model the density of the data.

        Args:
            df: Data frame containing the training data.
            diagnostics: Whether to compute and store diagnostic information.
        """
        df = categorize_non_numerics(df)
        self.schema = Schema.from_df(df)
        self.vp(f"Training a generative density model on {len(df)} samples")
        self.initial_density.train(df)
        sim_n = self.sim_size or compute_simulation_size(n_real=len(df))
        self.sim_rate = sim_n / len(df)
        self.vp(f"Simulating {sim_n} fake samples from the model and join it with the real data")
        sim_df = self.initial_density.rvs(sim_n)
        sim_df = self.schema(sim_df)
        xdf = pd.concat([df, sim_df], axis=0)
        assert isinstance(xdf, pd.DataFrame), "CadeData.X requires a pandas DataFrame"
        partially_synthetic_data = CadeData(
            X=xdf,
            y=np.concatenate([np.ones(df.shape[0]), np.zeros(sim_n)]),
        )
        self.vp("Train the classifier to distinguish real from fake")
        self.classifier.train(partially_synthetic_data)
        if diagnostics or self.verbose:
            self._diagnostics(partially_synthetic_data.X, partially_synthetic_data.y)
        if self.verbose:
            print(f"In-sample, the classifier had AUROC = {round(self.diagnostics[AUROC], 3)}")

    def density(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the density at new points.

        Apply equation 2.1 in https://pdfs.semanticscholar.org/e4e6/033069a8569ba16f64da3061538bcb90bec6.pdf

        Args:
            X: Data frame that can be coerced to match the schema of the training data.
        """
        X = self.schema(X)
        # Initial density estimate
        synthetic_dens = self.initial_density.density(X)
        # Classifier adjustment factor
        p_real = self.classifier.predict(X)
        odds_real = p_real / (1 - p_real)
        classifier_adjustment = self.sim_rate * odds_real
        return synthetic_dens * classifier_adjustment
