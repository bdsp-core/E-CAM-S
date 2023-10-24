import numpy as np
from mord import LogisticAT


class MyCalibrator:
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        
    def fit(self, X, y):
        yp = self._base_estimator_predict(X)
        self.recalibration_mapper = LogisticAT(alpha=0).fit(yp.reshape(-1,1), y)
        return self
    
    def _base_estimator_predict(self, X):
        K = len(self.base_estimator.classes_)
        yp = np.sum(self.base_estimator.predict_proba(X)*np.arange(K), axis=1)
        return yp
        
    def predict_proba(self, X):
        yp = self._base_estimator_predict(X)
        yp2 = self.recalibration_mapper.predict_proba(yp.reshape(-1,1))
        return yp2
        
    def predict(self, X):
        yp = self.predict_proba(X)
        K = yp.shape[1]
        yp = np.sum(yp*np.arange(K), axis=1)
        return yp

