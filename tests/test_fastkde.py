from fastkde import fastKDE
from scipy import stats


def test_fastkde_runs():
    gauss = stats.norm(-2, 4)
    data = gauss.rvs(size=100)
    _ = fastKDE.pdf(data)
