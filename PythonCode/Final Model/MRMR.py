# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:03:52 2021

@author: chris
"""
        
import pandas as pd
from sklearn.feature_selection import f_regression
        
def feat_MRMR(X_train, y_train, X_test, Xnames, num_feat):
    """
    https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/
    https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
    pip install git+https://github.com/smazzanti/mrmr    

    This is the FCQ variant, slightly modified to include Spearman's instead of Pearson's as described in (Ding & Peng et al. 2005)
    
    Parameters
    ----------
    X_train : array of features from training fold
    y_train : array of target from training fold
    X_test : array of features from test fold
    Xnames : names of features
    num_feat : # of features to be selected

    Returns
    -------
    X_train_select : selected features from training fold
    X_test_select : same features as X_train, based on metrics from training fold, but for the test fold (for running model on test fold)

    """
    F = pd.Series(f_regression(X_train, y_train)[0], index = Xnames)
    corr = pd.DataFrame(.00001, index = Xnames, columns = Xnames)

    feat_select = []
    feat_not_select = Xnames.to_list()
    df_X_train=pd.DataFrame(X_train)
    df_X_train.columns=Xnames

    df_X_test=pd.DataFrame(X_test)
    df_X_test.columns=Xnames

    for f in range(num_feat):
  
    
        if f > 0:
            last = feat_select[-1]
            corr.loc[feat_not_select, last] = df_X_train[feat_not_select].corrwith(df_X_train[last],method="spearman").abs().clip(.00001)
        
        # compute FCQ score for all the excluded features 
        score = F.loc[feat_not_select] / corr.loc[feat_not_select, feat_select].mean(axis = 1).fillna(.00001)
    
        # find best feature, add it to selected and remove it from not_selected
        best = score.index[score.argmax()]
        feat_select.append(best)
        feat_not_select.remove(best)
    
    X_train_select = df_X_train[feat_select].values
    X_test_select = df_X_test[feat_select].values
    Xnames_select= df_X_train[feat_select].columns
    
    return X_train_select, X_test_select, Xnames_select