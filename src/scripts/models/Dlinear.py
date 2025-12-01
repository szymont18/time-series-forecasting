from sktime.forecasting.ltsf import LTSFDLinearForecaster

class DLinearModel:
    def __init__(self, fh, **config):
        self.model = LTSFDLinearForecaster(**config)
        self.fh = fh

    def fit(self, train):
        self.model.fit(train)

    def __call__(self, *args, **kwargs):
        return self.model.predict(fh=self.fh)

    def eval(self):
        return

    def predict(self):
        return self.__call__()
