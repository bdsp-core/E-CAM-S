from itertools import combinations
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr, kendalltau
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder


class LTRPairwise(BaseEstimator, ClassifierMixin):
    """Learning to rank, pairwise approach
    For each pair A and B, learn a score so that A>B or A<B based on the ordering.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        It must be a classifier with a ``decision_function`` function.
    verbose : bool, optional, defaults to False
        Whether prints more information.
    """
    def __init__(self, estimator, verbose=False):
        super().__init__()
        self.estimator = estimator
        self.verbose = verbose
        self.classes_ = np.array([0,1,2,3,4,5,6,7])

    def _generate_pairs(self, X, y, sample_weight):
        X2 = []
        y2 = []
        sw2 = []
        for i, j in combinations(range(len(X)), 2):
            # if there is a tie, ignore it
            if y[i]==y[j]:
                continue
            X2.append( X[i]-X[j] )
            y2.append( 1 if y[i]>y[j] else 0 )
            if sample_weight is not None:
                sw2.append( (sample_weight[i]+sample_weight[j])/2 )

        if sample_weight is None:
            sw2 = None
        else:
            sw2 = np.array(sw2)

        return np.array(X2), np.array(y2), sw2

    def fit(self, X, y, sample_weight=None):
        self.fitted_ = False

        # generate pairs
        X2, y2, sw2 = self._generate_pairs(X, y, sample_weight)
        if self.verbose:
            print('Generated %d pairs from %d samples'%(len(X2), len(X)))

        # fit the model
        self.estimator.fit(X2, y2, sample_weight=sw2)

        # get the mean of z for each level of y
        self.label_encoder = LabelEncoder().fit(y)
        z = self.predict_z(X)
        self.z_means = np.array([z[y==cl].mean() for cl in self.label_encoder.classes_])
        self.fitted_ = True
        return self

    def predict_z(self, X):
        z = self.estimator.decision_function(X)
        return z

    def predict_proba(self, X):
        z = self.predict_z(X)
        dists = -np.abs(z.reshape(-1,1) - self.z_means)
        yp = softmax(dists, axis=1)
        return yp

    def predict(self, X):
        yp = self.predict_proba(X)
        yp1d = self.label_encoder.inverse_transform(np.argmax(yp, axis=1))
        return yp1d

    def score(self, X, y):
        yp = self.predict(X)
        return kendalltau(y, yp)[0]