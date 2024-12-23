import pandas as pd
import pytest

# import statement for IntCastingNaNError, raised by pandas astype:
from pandas.errors import IntCastingNaNError

from empdens.utils.pandahandler import Schema, categorize_non_numerics


def test_schema():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", None],
            "c": [1.0, 2.0, 3.0],
        }
    )
    target_df = pd.DataFrame(
        {
            "a": [4, 5, None],
            "b": ["y", None, "z"],
            "c": [1, 2, 3],
        }
    )
    schema = Schema.from_df(df)
    # expect this to fail because column `target_df.a` includes a null value
    with pytest.raises(IntCastingNaNError):
        coerced_df = schema(target_df)

    # but this should work ok
    target_df["a"] = [4, 5, 6]
    coerced_df = schema(target_df)
    pd.testing.assert_series_equal(df.dtypes, coerced_df.dtypes)

    # check that categoricals get encoded properly
    df = categorize_non_numerics(df)
    schema = Schema.from_df(df)
    coerced_df = schema(target_df)
    assert coerced_df["b"].cat.codes.to_list() == [
        1,  # 'y' gets coded to 1, not 0, even though it's the "first" category, consistent with df["b"]
        -1,  # None gets coded to -1 no matter what
        -1,  # The value "z", not seen in `df["b"]` also gets coded to None
    ]
