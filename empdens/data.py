"""Data structures."""

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


class CadeData(object):
    """A standardized data format for empdens.cade.Cade."""

    def __init__(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> None:
        """Initialize the CadeData object.

        Args:
            X: The feature matrix.
            y: The target vector.
        """
        assert len(y.shape) == 1
        assert len(y) == X.shape[0]
        self.X = X
        self.y = y


def load_Japanese_vowels_data() -> pd.DataFrame:
    """Data downloaded from http://odds.cs.stonybrook.edu/japanese-vowels-data/."""
    path = resources.files("empdens.resources.data").joinpath("japanese_vowels.csv")
    assert isinstance(path, Path)
    return pd.read_csv(path)


def load_SHAP_census_data():
    """Loads the 'adults' dataset available in SHAP.

    This borrows a few SHAP file parsing code snippets, https://github.com/slundberg/shap/blob/master/shap/datasets.py.
    """
    try:
        pass
    except Exception:
        raise Exception("Do `pip install shap` and try again")
    dtypes = [
        ("Age", "float32"),
        ("Workclass", "category"),
        ("fnlwgt", "float32"),
        ("Education", "category"),
        ("Education-Num", "float32"),
        ("Marital Status", "category"),
        ("Occupation", "category"),
        ("Relationship", "category"),
        ("Race", "category"),
        ("Sex", "category"),
        ("Capital Gain", "float32"),
        ("Capital Loss", "float32"),
        ("Hours per week", "float32"),
        ("Country", "category"),
        ("Target", "category"),
    ]
    df = pd.read_csv(
        "https://github.com/slundberg/shap/raw/master/data/adult.data",
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes),
    )
    df = df[df.Country == " United-States"].copy()
    df.drop(["Country", "Education", "fnlwgt"], axis=1, inplace=True)
    df.rename({"Target": "Income"}, axis=1, inplace=True)
    return df
