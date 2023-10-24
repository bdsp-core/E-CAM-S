# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:15:40 2021

@author: chris
"""

#from catboost import CatBoostRanker, Pool, MetricVisualizer
from copy import deepcopy
import numpy as np
import os
import pandas as pd

from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score
from learning_to_rank_ICANSYeti import LTRPairwiseCat

from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class OrdinalClassifier():
    
    def __init__(self, clf):
        self.clf = clf

        self.clfs = {}
    
    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
    
    def predict_proba(self, X):
        clfs_predict = {k:self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i,y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                 predicted.append(clfs_predict[y-1][:,1] - clfs_predict[y][:,1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

#%%
y = keep['ICANS'].apply(int).values

#ICANS score
sids=keep["SID"].apply(int).values  #SIDs
Features=keep.iloc[:,0:88]
#%%

Xnames = Features.columns
X = Features.values


def get_perf(y, yp):
    perf = spearmanr(y, yp).correlation
    return perf

coefs = []
 
print('X shape', X.shape)
isXnan = np.isnan(X).any()
print('X is nan',isXnan)
  
cv_tr_score = []
cv_te_score = []

#Split data into train and test folds, stratifying outcomes y, and insuring no groups are split across train and test fold

        
for k, (trid, teid) in enumerate(stratified_group_k_fold(X, y, sids, cvf, seed=random_state)):
    if k >0:
        break
    else:
        X_train = X[trid]
        
        #Scale after split, save parameters
        pt=  PowerTransformer()
        X_train= pt.fit_transform(X_train)
        X_test = X[teid] 
        X_test=pt.transform(X_test) #scale test set using training scaling parameters
        
        y_train = y[trid]
        y_test = y[teid]
        
        #Group info (ie patient)
        sids_test = sids[teid]
        sids_train=sids[trid]
        
        mses = []
        mses_inner_loop = []
        alphas = []
        ytes = []        # y-test values
        yptes = []       # predicted y values 
        yptes_z = []
        sids_tes = []
        bestparam = []

        clf = OrdinalClassifier(xgb.XGBClassifier( learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=2))
        
        param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
        gsearch1 = GridSearchCV(estimator = OrdinalClassifier(xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5))
        gsearch1.fit(X_train,y_train)
        
        
        
        clf.fit(X_train,y_train)
        ypte = clf.predict_proba(X_test)  # this is the calibrated probability of each CAM-S level (8 numbers) for each patient
        ypte = np.sum(ypte*np.arange(5), axis=1)  #

        yptr = clf.predict_proba(X_train)  # this is the calibrated probability of each CAM-S level (8 numbers) for each patient
        yptr = np.sum(yptr*np.arange(5), axis=1)
       #  LTR = LTRPairwiseICANS(estimator=LogisticRegression(penalty='l1',solver='liblinear',max_iter=100,class_weight="balanced"))  #lasso, will try with ‘elasticnet’ too
            
#     random_seed=42,
#     logging_level='Silent'
# )
        model.fit(X_train,y_train)
        p = model.predict_proba(X_train)[1]
        p_means = np.array([p[y_train==cl].mean() for cl in model.label_encoder.classes_])
       
            # Run LTR model

#%%            
            
                                                                           
            LTR_cv = GridSearchCV(LTR, 
                                    {'estimator__C': np.logspace(-3,1,6) #{'estimator__l1_ratio':np.arange(0.0,1,0.1) ## #, #try np.logspace(-3,1,6),  originally [0.01,0.1,1,10,100]
                                    },
                                    n_jobs=4, cv=stratified_group_k_fold(X_train, y_train, sids_train, cvf, seed=random_state))#group_kfold.split(X_train, y_train, sids_train)) # #could add scoring= make_scorer(lambda y,yp:spearmanr(y,yp).correlation)
            LTR_cv.fit(X_train, y_train)
            #ypte = LTR_cv.best_estimator_.predict(X_test)  # this is one integer (0,1,2,...7) for each patient
            ypte_z = LTR_cv.best_estimator_.predict_z(X_test) # this is a unbounded float number (-inf, +inf) for each patient
            #best_estimator extracts the best hyperparameter from GridSearchCV
            
            # calibrate
            LTR_cv = MyCalibrator(LTR_cv)
            #import pdb;pdb.set_trace()
            LTR_cv.fit(X_train, y_train)
 