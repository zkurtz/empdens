"""Evaluation of binary predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn import metrics


@dataclass
class Binary:
    """A collection of metrics for the strength of association between two vectors in [0,1]."""

    truth: np.ndarray
    pred: np.ndarray

    @property
    def AUROC(self):
        """Area under the receiver-operator characteristic curve."""
        value = metrics.roc_auc_score(y_true=self.truth, y_score=self.pred)
        return float(value)

    @property
    def rank_order_corr(self) -> float:
        """Rank-order correlation."""
        value = pd.Series(self.truth).corr(pd.Series(self.pred), method="spearman")
        return float(value)
