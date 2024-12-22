import numpy as np
import pandas as pd

from empdens import simulators
from empdens.cade import Cade
from empdens.classifiers.lightgbm import Lgbm
from empdens.models import JointDensity


def test_cade_default():
    x = Cade()
    assert x is not None


def test_cade():
    # Compile a Cade class with all defaults
    np.random.seed(0)
    N = 100
    sz = simulators.Zena()
    data = sz.rvs(100)
    cade = Cade(initial_density=JointDensity(), classifier=Lgbm(), sim_size=N)
    cade.train(pd.DataFrame(data), diagnostics=True)
    diagnostics = cade.diagnostics

    assert all([x in diagnostics.keys() for x in ["val_df", "auc"]])
    df = diagnostics["val_df"]
    assert df.shape[0] == 2 * N
    auc = diagnostics["auc"]
    assert isinstance(auc, float)
    assert auc >= 0
    assert auc <= 1

    dens = cade.density(data.iloc[:3])
    assert len(dens) == 3
    assert isinstance(dens, np.ndarray)
    assert dens.dtype == "float64"
