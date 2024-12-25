"""Execute the code used for notebooks/demo.ipynb."""

import numpy as np
import pandas as pd

from empdens import cade, classifiers, evaluation, models, simulators
from empdens.wrappers.fast_kde import FastKDE
from empdens.wrappers.sklearn_isolation_forest import SklearnIsolationForest
from empdens.wrappers.sklearn_kde import SklearnKDE


def _build_random_data():
    np.random.seed(0)
    sz = simulators.Zena()
    df = sz.rvs(1000)
    return df


class Estimators:
    def __init__(self):
        self.df = _build_random_data()
        self.simulation = simulators.Zena()
        lgb = classifiers.lightgbm.Lgbm()  # pyright: ignore
        self.estimators = [
            FastKDE(),
            SklearnKDE(),
            SklearnIsolationForest(),
            cade.Cade(initial_density=models.JointDensity(), classifier=lgb),
        ]

    def train(self):
        for estimator in self.estimators:
            estimator.train(self.df)

    def densities(self) -> dict[str, pd.Series]:
        return {type(estimator).__name__: estimator.density(self.df) for estimator in self.estimators}


def test_zena():
    df = _build_random_data()
    expected_head_df = pd.DataFrame(
        {
            "gaussian": {
                0: 0.14858811631077062,
                1: 0.5877388267149037,
                2: 0.2839651902198238,
                3: 0.13886071758977125,
                4: -0.15920594260813145,
            },
            "triangular": {
                0: 1.0858219615277211,
                1: 0.015133715039288398,
                2: 0.8280045494327123,
                3: 1.3810291906490217,
                4: 0.06670473498094898,
            },
        }
    )
    df_head = df.head()
    pd.testing.assert_frame_equal(df_head, expected_head_df)


def test_cade():
    df = _build_random_data()
    lgb = classifiers.lightgbm.Lgbm()  # pyright: ignore
    cc = cade.Cade(initial_density=models.JointDensity(), classifier=lgb)
    cc.train(df, diagnostics=True)
    assert cc.diagnostics["auc"] == 0.9112022454142947


# @pytest.mark.skip(reason="To much refactoring happening now")
def test_evaluation():
    estimators = Estimators()
    estimators.train()
    generative_density = estimators.simulation.density
    df = estimators.df
    ev = evaluation.Evaluation(estimators=estimators.densities(), truth=generative_density(df))
    expected_output = pd.DataFrame(
        {
            "FastKDE": {
                "mean_absolute_error": 0.017839340471659436,
                "mean_squared_error": 0.0007697280337532455,
                "rank-order correlation": 0.9525405805674115,
                "pearson correlation": 0.9393207787944629,
                "mean density": 0.11658864131600007,
            },
            "SklearnKDE": {
                "mean_absolute_error": 0.06530592283410162,
                "mean_squared_error": 0.007369769510032537,
                "rank-order correlation": 0.7834176634176635,
                "pearson correlation": 0.7798812303705205,
                "mean density": 0.06428516722141582,
            },
            "SklearnIsolationForest": {
                "mean_absolute_error": 0.3848338692422281,
                "mean_squared_error": 0.15059037577956008,
                "rank-order correlation": 0.7454989934989935,
                "pearson correlation": 0.7091455426547734,
                "mean density": 0.5125398482700011,
            },
            "Cade": {
                "mean_absolute_error": 0.6859270924805501,
                "mean_squared_error": 0.9127381923442783,
                "rank-order correlation": 0.8645335036691795,
                "pearson correlation": 0.8428912746657217,
                "mean density": 0.8112451568849589,
            },
        }
    )
    df = ev.evaluate()
    pd.testing.assert_frame_equal(
        df[expected_output.columns],
        expected_output,
        # TODO: set all the random seeds properly and try to get this more precise
        check_exact=False,
        atol=1e-1,
    )
