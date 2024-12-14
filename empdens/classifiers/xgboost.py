import copy
import os
import pickle
import tempfile
from time import time

import pandas as pd
import xgboost as xgb
from psutil import cpu_count

from empdens.classifiers.base import AbstractLearner


class Xgbm(AbstractLearner):
    def __init__(self, params=None, verbose=False):
        super().__init__(params, verbose)
        self.nround = self.params.pop("num_boost_round")

    def default_params(self):
        return {
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "binary",
            "learning_rate": 0.1,
            "max_depth": 6,
            "verbose": -1,
            "nrounds": 60,
            "nthreads": cpu_count(logical=False),
        }

    def as_xgb_data(self, data):
        self._parse_categoricals()
        return xgb.Dataset(data.X, data.y)

    def train(self, data):
        """:param data: a empdens.data.Data instance"""
        t0 = time()
        ld = self.as_lgb_data(data)
        self.bst = lgb.train(
            params=copy.deepcopy(self.params), train_set=ld, num_boost_round=self.nround, verbose_eval=False
        )
        tdiff = str(round(time() - t0))
        self.vp("Xgboost training took " + tdiff + " seconds")

    def predict(self, X):
        return self.bst.predict(X)

    def freeze(self):
        """Attach self.bst as a binary attribute.

        This is necessary to be able to preserve by-reference internals during a
        serialization-unserialization cycle
        """
        assert self.bst is not None
        _, filename = tempfile.mkstemp()
        self.bst.save_model(filename)
        with open(filename, "rb") as file:
            self.bst_binary = file.read()
        os.remove(filename)

    def thaw(self):
        """Unserialize self.bst_binary."""
        assert hasattr(self, "bst_binary")
        assert self.bst_binary is not None
        self.bst = pickle.loads(self.bst_binary)

    def importance(self):
        return (
            pd.DataFrame({"feature": self.features, "gain": self.bst.feature_importance(importance_type="gain")})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
