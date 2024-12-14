import pytest
from scipy import stats

fastkde = pytest.importorskip("fastkde")
from fastkde import fastKDE


def test_fastkde_runs():
    gauss = stats.norm(-2, 4)
    data = gauss.rvs(size=100)
    _ = fastKDE.pdf(data)
