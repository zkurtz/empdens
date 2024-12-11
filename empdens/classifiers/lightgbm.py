import copy
import os
import pandas as pd
import pickle
from psutil import cpu_count
import tempfile
from time import time
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import lightgbm as lgb
    except:
        pass

from .base import AbstractLearner

def assert_lightgbm_installed():
    try:
        lgb
    except:
        raise Exception("empdens.classifiers.lightgbm.Lgbm requires lightgbm to be installed, but it is not")

class Lgbm(AbstractLearner):
    def __init__(self, params=None, categorical_features=None, verbose=False):
        super().__init__(params, verbose)
        assert_lightgbm_installed()
        self.nround = self.params.pop('num_boost_round')
        self.categorical_features = categorical_features

    def default_params(self):
        return {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'xentropy',
            'learning_rate': 0.07,
            'num_leaves': 20,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'num_boost_round': 50,
            'num_threads': cpu_count(logical=False)
        }

    def _parse_categoricals(self):
        if self.categorical_features is None:
            self.categoricals='auto'
        else:
            self.categoricals = self.categorical_features
            assert all([c in self.features for c in self.categoricals])

    def as_lgb_data(self, data):
        self.features = data.X.columns.tolist()
        self._parse_categoricals()
        return lgb.Dataset(
            data.X,
            data.y,
            feature_name=self.features,
            categorical_feature=self.categoricals
        )

    def train(self, data):
        '''
        :param data: a empdens.data.Data instance
        '''
        t0 = time()
        ld = self.as_lgb_data(data)
        self.bst = lgb.train(
            params=copy.deepcopy(self.params),
            train_set=ld,
            num_boost_round=self.nround,
            verbose_eval=False,
            feature_name=self.features,
            categorical_feature=self.categoricals
        )
        tdiff = str(round(time() - t0))
        self.vp('LightGBM training took ' + tdiff + ' seconds')

    def predict(self, X):
        return self.bst.predict(X)

    def freeze(self):
        ''' Attach self.bst as a binary attribute

        This is necessary to be able to preserve by-reference internals during a
        serialization-unserialization cycle
        '''
        assert self.bst is not None
        _, filename = tempfile.mkstemp()
        self.bst.save_model(filename)
        with open(filename, 'rb') as file:
            self.bst_binary = file.read()
        os.remove(filename)

    def thaw(self):
        ''' Unserialize self.bst_binary '''
        assert hasattr(self, 'bst_binary')
        assert self.bst_binary is not None
        self.bst = pickle.loads(self.bst_binary)

    def importance(self):
        return pd.DataFrame({
            'feature': self.features,
            'gain': self.bst.feature_importance(importance_type='gain')
        }).sort_values('gain', ascending=False
        ).reset_index(drop=True)

