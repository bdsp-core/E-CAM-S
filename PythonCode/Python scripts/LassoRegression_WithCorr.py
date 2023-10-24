import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, check_X_y
from scipy.stats import spearmanr

class LassoRegressionWithCorr(BaseEstimator, RegressorMixin):
    """Linear ordinal regression

    Parameters
    ----------
    alpha1 : float, optional
        Constant that multiplies the penalty term of the coefficients. Defaults to None, which is 0.
    alpha2 : float, optional
        Constant that multiplies the penalty term of the correlation term. Defaults to None, which is 0.
    max_iter : int, optional
        The maximum number of iterations
    verbose : bool, optional
        If print the optimization result or not.
    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula).
    intercept_ : float
        independent term in decision function.
    """
    def __init__(self, alpha1=None, alpha2=None, max_iter=1000, verbose=False, random_state=None):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state
        
    def fit(self, X, y, sample_weight=None):
        self.fitted_ = False
        self.opt_res = None
        
        # do some checks
        X, y = check_X_y(X, y)
        N, self.n_feat = X.shape

        if self.alpha1<0:
            raise ValueError('alpha1 is %f. But it must be positive.'%self.alpha1)
        if self.alpha2<0:
            raise ValueError('alpha2 is %f. But it must be positive.'%self.alpha2)

        # define loss function; MSE + alpha1*|w| + alpha2*correlation(y, yp-y)
        def loss(params):
            w = params[:self.n_feat]
            b = params[self.n_feat]
            yp = np.dot(X, w)+b
            
            # loss
            mse = np.mean((yp - y)**2) 
            w_l1 = np.sum(np.abs(w)) 
            corr = np.abs(spearmanr(y, yp-y)[0])
            loss = mse + self.alpha1*w_l1 + self.alpha2*corr
            return loss

        w0 = np.random.randn(self.n_feat)*0.001
        b0 = 0
        params0 = np.r_[w0, b0]

        self.opt_res = minimize(loss, params0,# method='Nelder-Mead',
                        options={'maxiter':self.max_iter, 'disp':self.verbose})
        
        self.coef_ = self.opt_res.x[:self.n_feat]
        self.intercept_ = self.opt_res.x[-1]
        
        self.fitted_ = True
        return self

    def predict(self, X):
        yp = np.dot(X, self.coef_) + self.intercept_
        return yp