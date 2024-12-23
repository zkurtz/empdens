"""Utilities for handling pandas series and data frames."""

from dataclasses import dataclass
from functools import cached_property
from typing import Hashable

import pandas as pd
from typing_extensions import Self


def _get_categoricals(types_: pd.Series) -> list[Hashable]:
    """Return the names of categorical columns."""
    return [col for col, dtype in types_.items() if isinstance(dtype, pd.CategoricalDtype)]


def _get_numerics(types_: pd.Series) -> list[Hashable]:
    """Return the names of numeric columns."""
    return [col for col, dtype in types_.items() if pd.api.types.is_numeric_dtype(dtype)]


def categorize_non_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize columns that are neither categorical nor numeric."""
    categoricals = _get_categoricals(df.dtypes)
    numerics = _get_numerics(df.dtypes)
    num_or_cat = set(categoricals + numerics)
    others = [col for col in df if col not in num_or_cat]
    df[others] = df[others].astype("category")
    return df


@dataclass(frozen=True)
class Schema:
    """Managing a data frame's schema information, including tools of coercion."""

    types_: pd.Series
    categorical_encodings: dict[Hashable, pd.Index]

    def __post_init__(self):
        """Run consistency checks."""
        categoricals = set(self.categoricals)
        numerics = set(self.numerics)
        others = set(self.others)
        assert not categoricals & numerics, "Categorical and numeric columns overlap."
        assert not categoricals & others, "Categorical and other columns overlap."
        assert not numerics & others, "Numeric and other columns overlap."
        assert set(self.types_.index) == categoricals | numerics | others, "Column names are inconsistent."

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """Create a ColumnTypes object from a data frame."""
        categoricals = _get_categoricals(df.dtypes)
        return cls(
            types_=df.dtypes,
            categorical_encodings={col: df[col].cat.categories for col in categoricals},
        )

    @cached_property
    def categoricals(self) -> list[Hashable]:
        """Return the names of categorical columns."""
        return list(self.categorical_encodings) if self.categorical_encodings else []

    @cached_property
    def numerics(self) -> list[Hashable]:
        """Return the names of numeric columns."""
        return [col for col, dtype in self.types_.items() if pd.api.types.is_numeric_dtype(dtype)]

    @cached_property
    def others(self) -> list[Hashable]:
        """Return the names of columns that are neither categorical nor numeric."""
        numerics = self.numerics
        categoricals = self.categoricals
        categorical_or_numeric = set(categoricals + numerics)
        return [col for col in self.types_.index if col not in categorical_or_numeric]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce the data frame to the schema."""
        df = df.copy()
        other_types = self.types_.loc[pd.Index(self.others)]
        df[self.others] = df[self.others].astype(other_types)

        numeric_types = self.types_.loc[pd.Index(self.numerics)]
        df[self.numerics] = df[self.numerics].astype(numeric_types)

        for col in self.categoricals:
            df[col] = df[col].astype("category")
            df[col] = df[col].cat.set_categories(self.categorical_encodings[col])

        if not df.dtypes.equals(self.types_):
            raise ValueError("The schema coercion failed.")

        return df
