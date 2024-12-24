import numpy as np
import pandas as pd

from empdens import simulators
from empdens.simulators import BartSimpson


def test_zena():
    np.random.seed(0)
    N = 100
    sz = simulators.Zena()
    data = sz.rvs(N)
    assert isinstance(data, pd.DataFrame)
    assert data.shape[0] == N
    assert data.shape[1] == 2
    assert all(data.columns == ["gaussian", "triangular"])


def test_bart_simpson():
    bart_simpson = BartSimpson()
    bart_simpson.plot()
