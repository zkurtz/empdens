from abc import ABC, abstractmethod


class AbstractLearner(ABC):
    def __init__(self, params=None, verbose=False):
        self.params = self.default_params()
        if params is not None:
            self.params.update(copy.deepcopy(params))
        self.verbose = verbose

    @abstractmethod
    def default_params(self):
        pass

    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def vp(self, string):
        if self.verbose:
            print(string)
