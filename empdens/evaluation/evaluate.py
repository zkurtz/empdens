"""Evaluation of density estimators."""

import numpy as np
import pandas as pd


class Evaluation(object):
    """A collection of metrics of density estimation performance."""

    def __init__(self, estimators, truth=None):
        """Initialize the Evaluation object.

        Args:
            estimators: A dictionary of estimator predictions.
            truth: The true density values.
        """
        for est in estimators:
            assert isinstance(estimators[est], np.ndarray)
        self.estimators = estimators
        self.truth = truth
        if truth is not None:
            assert isinstance(truth, np.ndarray)
        self.metrics = {
            "mean_absolute_error": self.mean_absolute_error,
            "mean_squared_error": self.mean_squared_error,
            "rank-order correlation": self.rank_order_correlation,
            "pearson correlation": self.pearson_correlation,
            "mean density": self.mean_density,
        }

    def _assert_truth_available(self):
        """Raise an exception if truth is not available."""
        if self.truth is None:
            raise Exception("rank-order correlation can't be computed since you did not provide anything for `truth`")

    def rank_order_correlation(self, pred):
        """Spearman (i.e. rank-order) correlation of prediction against truth.

        Maximize me
        """
        self._assert_truth_available()
        s_pred = pd.Series(pred)
        t_pred = pd.Series(self.truth)
        return s_pred.corr(t_pred, method="spearman")

    def pearson_correlation(self, pred):
        """Spearman (i.e. rank-order) correlation of prediction against truth.

        Maximize me
        """
        self._assert_truth_available()
        s_pred = pd.Series(pred)
        t_pred = pd.Series(self.truth)
        return s_pred.corr(t_pred)

    def mean_absolute_error(self, pred):
        """Direct estimate of expectation of L1 loss."""
        self._assert_truth_available()
        return np.absolute(pred - self.truth).mean()

    def mean_squared_error(self, pred):
        """Direct estimate of expectation of L2 loss."""
        self._assert_truth_available()
        squared_errors = (pred - self.truth) ** 2
        return squared_errors.mean()

    def mean_density(self, pred):
        """The mean density; this is exp(-deviance).

        This is a sensible metric for some (but not all) density estimators
        """
        return np.mean(pred)

    def evaluate_estimator(self, name):
        """Evaluate the estimator with the given name."""
        est = self.estimators[name]
        return [fun(est) for name, fun in self.metrics.items()]

    def evaluate(self):
        """Evaluate all estimators."""
        estimators = self.estimators.keys()
        return pd.DataFrame({e: self.evaluate_estimator(e) for e in estimators}, index=pd.Index(self.metrics))
