import pytest


@pytest.mark.filterwarnings("ignore:numpy.ufunc size changed")
def test_zena():
    import numpy as np
    import pandas as pd

    from empdens import simulators

    np.random.seed(0)
    N = 100
    sz = simulators.bivariate.Zena()
    data = sz.rvs(N)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] == N
    assert data.shape[1] == 2
    assert all(data.columns == ["gaussian", "triangular"])
