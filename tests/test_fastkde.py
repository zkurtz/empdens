import pandas as pd
from scipy import stats

from empdens.wrappers.fast_kde import FastKDE


def test_fastkde_runs():
    gauss = stats.norm(-2, 4)
    data = gauss.rvs(size=100)
    df = pd.DataFrame(data, columns=pd.Index(["gauss"]))
    model = FastKDE()
    model.train(df)
    true_dens = pd.Series(gauss.pdf(df["gauss"]))  # pyright: ignore[reportAttributeAccessIssue]
    est_dens = pd.Series(model.density(df))
    assert true_dens.corr(est_dens) > 0.5
