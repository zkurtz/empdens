"""Evaluation of binary predictions."""

import numpy as np
from sklearn import metrics


class Binary(object):
    """A collection of metrics for the strength of association between two vectors in [0,1]."""

    def __init__(self, truth, pred):
        """Initialize the Binary object."""
        assert len(truth) == len(pred)
        assert isinstance(truth, np.ndarray)
        assert isinstance(pred, np.ndarray)
        self.truth = truth
        self.pred = pred
        self.metrics = {
            "auroc": self.AUROC
            # 'rank-order correlation': self.rank_order_correlation,
            # 'pearson correlation': self.pearson_correlation
        }

    def AUROC(self):
        """Area under the receiver-operator characteristic curve."""
        return metrics.roc_auc_score(y_true=self.truth, y_score=self.pred)
