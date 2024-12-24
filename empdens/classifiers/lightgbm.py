"""LightGBM classifier."""

import copy
import os
import pickle
import tempfile
from time import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from psutil import cpu_count

from empdens.classifiers.base import AbstractLearner


class Lgbm(AbstractLearner):
    """LightGBM classifier."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> None:
        """Initializes the LightGBM model."""
        super().__init__(params, verbose)
        self.nround = self.params.pop("num_boost_round")

    def default_params(self):
        """Returns the default parameters for the LightGBM model.

        Returns:
            dict: A dictionary containing the default parameters.
        """
        return {
            "task": "train",
            "boosting_type": "gbdt",
            "objective": "xentropy",
            "learning_rate": 0.07,
            "num_leaves": 20,
            "min_data_in_leaf": 20,
            "verbose": -1,
            "num_boost_round": 50,
            "num_threads": cpu_count(logical=False),
        }

    def _parse_categoricals(self):
        """Parses the categorical features for the LightGBM model."""
        if self.categorical_features is None:
            self.categoricals = "auto"
        else:
            self.categoricals = self.categorical_features
            assert all([c in self.features for c in self.categoricals])

    def as_lgb_data(self, data):
        """Converts the input data to a LightGBM Dataset.

        Args:
            data (empdens.data.Data): The input data.

        Returns:
            lgb.Dataset: The LightGBM Dataset.
        """
        self.features = data.X.columns.tolist()
        self.categorical_features = data.X.select_dtypes(include=["category"]).columns.tolist()
        self._parse_categoricals()
        return lgb.Dataset(
            data.X,
            data.y,
            feature_name=self.features,
            categorical_feature=self.categoricals,  # pyright: ignore[reportArgumentType]
        )

    def train(self, data):
        """Trains the LightGBM model.

        Args:
            data (empdens.data.Data): The input data.
        """
        t0 = time()
        ld = self.as_lgb_data(data)
        self.bst = lgb.train(
            params=copy.deepcopy(self.params),
            train_set=ld,
            num_boost_round=self.nround,
        )
        tdiff = str(round(time() - t0))
        self.vp("LightGBM training took " + tdiff + " seconds")

    def predict(self, X) -> np.ndarray:
        """Predicts the target values using the trained LightGBM model.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            np.ndarray: The predicted target values.
        """
        preds = self.bst.predict(X)
        assert isinstance(preds, np.ndarray), "LightGBM predictions must be a numpy array"
        return preds

    def freeze(self):
        """Serializes the trained LightGBM model to a binary attribute."""
        assert self.bst is not None
        _, filename = tempfile.mkstemp()
        self.bst.save_model(filename)
        with open(filename, "rb") as file:
            self.bst_binary = file.read()
        os.remove(filename)

    def thaw(self):
        """Deserializes the LightGBM model from the binary attribute."""
        assert hasattr(self, "bst_binary")
        assert self.bst_binary is not None
        self.bst = pickle.loads(self.bst_binary)

    def importance(self):
        """Returns the feature importance of the trained LightGBM model.

        Returns:
            pd.DataFrame: A DataFrame containing the feature importance.
        """
        return (
            pd.DataFrame({"feature": self.features, "gain": self.bst.feature_importance(importance_type="gain")})
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
