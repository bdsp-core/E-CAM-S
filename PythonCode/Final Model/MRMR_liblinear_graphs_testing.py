# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:04:36 2021

@author: chris
"""



# Library/package imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from stratgroup import stratified_group_k_fold
from MRMR import feat_MRMR

from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import os
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

from learning_to_rank_ICANS import LTRPairwiseICANS
from mycalibrator import MyCalibrator
from sklearn.feature_selection import f_regression
from time import process_time
import winsound
import random 
duration = 1000  # milliseconds
freq = 440  # Hz


#%%  KEEPER
Features=pd.read_excel(r"E:\Features\No84.xlsx")

#%%
keep=keep.reset_index()

keep=keep.iloc[:,2:]

#%%
y = keep['ICANS'].apply(int).values

#ICANS score
sids=keep["SID"].apply(int).values  
files=keep["File"].values
Features2=keep.iloc[:,3:96]
Features2=Features2.join(keep.iloc[:,1])


#%%

Xnames = Features2.columns
X = Features2.values

print(np.mean(np.isinf(X))) 

# shape
print(X.shape) #check dimensions
print(y.shape)
#print(Xnames)
print(Xnames.shape)
print(sids.shape)

#%%
import warnings
warnings.filterwarnings("ignore")


def myfit(X, y, sids, files, num_feat, cvf, random_state=42):
    
    # print('X shape', X.shape)
  #  isXnan = np.isnan(X).any()
    # print('X is nan',isXnan)
    

    cv_tr_score = np.empty((cvf,2))
    cv_te_score = np.empty((cvf,2))
    mses = []
    
    ytes = []        # y-test values
    yptes = []       # predicted y values 
    yptes_z = []
    sids_tes = []
    files_tes=[]
    bestparam = []

    #Split dataset, stratifying target among folds, and insuring no group is split between test and train fold

    for k, (trid, teid) in enumerate(stratified_group_k_fold(X, y, sids, cvf, seed=random_state)):
        if k ==1:
            break
        
        X_train = X[trid]
        
        ###FEATURE SCALING
        pt=  PowerTransformer()
        X_train= pt.fit_transform(X_train)

        X_test = X[teid] 
        X_test=pt.transform(X_test) #scaled using parameters from training fold, not refitting
        
        y_train = y[trid]
        y_test = y[teid]
        
        sids_test = sids[teid]
        sids_train=sids[trid]
        files_test=files[teid] #save this info for running correlations later, has date info
        
        ###FEATURE SELECTION via MRMR
        ##https://eng.uber.com/research/maximum-relevance-and-minimum-redundancy-feature-selection-methods-for-a-marketing-machine-learning-platform/
        ##https://towardsdatascience.com/mrmr-explained-exactly-how-you-wished-someone-explained-to-you-9cf4ed27458b
        
        
        X_train, X_test, __ = feat_MRMR(X_train, y_train, X_test, Xnames, num_feat)
        # print('after MRMR selection', X_train.shape[1])
        
        ###LTR model
        LTR = LTRPairwiseICANS(estimator=LogisticRegression(penalty='l1',solver='liblinear',max_iter=1000,class_weight="balanced"))  #lasso, will try with ‘elasticnet’ too
        
        ###Hyperparameter tuning 
        LTR_cv = GridSearchCV(LTR, 
                                {'estimator__C': np.logspace(-3,1,6)# {'estimator__C': np.logspace(-3,1,6) ## #, #try np.logspace(-3,1,6),  originally [0.01,0.1,1,10,100]
                                },
                                n_jobs=-1, cv=stratified_group_k_fold(X_train, y_train, sids_train, cvf, seed=random_state))#group_kfold.split(X_train, y_train, sids_train)) # #could add scoring= make_scorer(lambda y,yp:spearmanr(y,yp).correlation)
        LTR_cv.fit(X_train, y_train)
        ypte_z = LTR_cv.best_estimator_.predict_z(X_test) # this is a unbounded float number (-inf, +inf) for each patient
        #best_estimator extracts the best hyperparameter from GridSearchCV
        
        # calibrate
        LTR_cv = MyCalibrator(LTR_cv)
        LTR_cv.fit(X_train, y_train)
        
        # predict test
        ypte = LTR_cv.predict_proba(X_test)  # this is the calibrated probability of each ICANS for each patient
        ypte = np.sum(ypte*np.arange(5), axis=1)  # convert ypte into a float number bounded between [0,4] for each patient
        
        #predict train (so can assess for overfitting)
        yptr = LTR_cv.predict_proba(X_train)  # this is the calibrated probability of each CAM-S level (8 numbers) for each patient
        yptr = np.sum(yptr*np.arange(5), axis=1) 
                
        # evaluate
        cv_tr_score[k,:]= spearmanr(y_train, yptr)
        cv_te_score[k,:]= spearmanr(y_test, ypte) #also includes p-value
        score = np.mean((y_test - ypte)**2)
        ytes.extend(y_test)
        yptes.extend(ypte)
        yptes_z.extend(ypte_z)
        sids_tes.extend(sids_test)
        files_tes.extend(files_test)
        mses.append(score)
        bestparam.append(LTR_cv.base_estimator.best_params_['estimator__C'])#'estimator__C']) #estimator__l1_ratio'])  #)estimator__C']) #'
            
    mean_mse = np.median(mses)        
    mean_C = np.median(bestparam)    


    #scale data 
    pt= PowerTransformer()
    X=pt.fit_transform(X)

    ###
    #Feature selection 
    X2, __, Xnames2=feat_MRMR(X,y,X,Xnames,num_feat)
        
    LTR2 =LTRPairwiseICANS (estimator=LogisticRegression(penalty='l1',solver='liblinear',C=mean_C,max_iter=1000,class_weight="balanced"))
    LTR2.fit(X2, y)
    
        
    # calibrate
    LTR2 = MyCalibrator(LTR2)
    LTR2.fit(X2, y)
        
    # print('mses',mses)                     #mses for all folds
    # print('mean mse',mean_mse)             #mean mse
    # print('mean l1_ratio', mean_l1ratio)
    # print('LTR coefficients',LTR2.base_estimator.estimator.coef_)
    
    return LTR2.base_estimator.estimator.coef_, mean_mse, yptes, yptes_z, ytes, sids_tes, files_tes, Xnames2, cv_tr_score, cv_te_score

#######

# do bootstrap to get the confidence interval around the coefs 

n_iterations = 1#1200
n_folds=5
num_feat=20
random.seed(42)

N = len(X) 
Nrange=np.arange(0,len(X))

# run bootstrap

#Initialize empty dataframe and numpy arrays for bootstrapping
df_coef = pd.DataFrame()
cv_te_score_tot= np.empty((n_iterations,n_folds))
cv_tr_score_tot=np.empty((n_iterations,n_folds))
ytes_tot=np.empty((n_iterations,N)) ##bootstraps x #EEGs (1000x431)
yptes_tot=np.empty((n_iterations,N))
yptes_z_tot=np.empty((n_iterations,N))
mse_tot=np.empty((n_iterations,1))

elapsed_time=0
##BOOTSTRAPPING LOOP
count=0

# randrange gives you an integral value


for i in range(n_iterations):

    #Tracking progress
    print("Iteration ", i, "out of", n_iterations )
    if count==1000:
        break
    
    
    t1_start = process_time() 
    
    if i==0:
        Xbt = X
        ybt = y
        sidsbt = sids
        filesbt= files
    else:
        # use the bootstrapping ids to index X and y
        # generate bootstrapping ids
        ids=random.choices(Nrange,k=N)
        Xbt = X[ids]
        ybt = y[ids]
        sidsbt = sids[ids]
        filesbt=files[ids]

    try:
    # fit the model using Xbt and ybt
        coef, mse, yptes_, yptes_z_, ytes_, sids_tes_, files_tes_, Xnames2_, cv_tr_score_, cv_te_score_  = myfit(Xbt, ybt, sidsbt,filesbt, num_feat,n_folds)
 
        if i==0:
            yptes = yptes_      #predicted CAMS
            ytes = ytes_        #true test values
            yptes_z = yptes_z_  #predicted z-scores CAMS
            sids_tes = sids_tes_
            files_tes=files_tes_
            Xnames2 = Xnames2_
            cv_tr_score=cv_tr_score_
            cv_te_score=cv_te_score_
            # print("cv_te_score: ", cv_te_score)
            # print("cv_tr_score: ", cv_tr_score)
    
        #Features selected via MRMR can vary, so need to retain name
        df_c= pd.DataFrame(coef)
        df_c.columns=Xnames2_
        df_coef=df_coef.append(df_c)
        
        #Save performance metrics for each bootstrap into array *n_iterations x folds
        cv_te_score_tot[count,:]=cv_te_score_[:,0]
        cv_tr_score_tot[count,:]=cv_tr_score_[:,0]
        ytes_tot[count,:]=ytes_
        yptes_tot[count,:]=yptes_
        yptes_z_tot[count,:]=yptes_z_
        mse_tot[count]=mse
        
        
        t1_stop = process_time()
     #every 10th loop, excluding the first loop 
        loop_time=t1_stop-t1_start
        elapsed_time= elapsed_time+loop_time
        remain_time = (elapsed_time/(count+1))*(n_iterations-(count+1))/60
        count = count+1
        if count%10==0:
            print(elapsed_time/60, " minutes have passed")
            print(remain_time, " minutes approxiately remaining")
    except:
        print("Error, skipping bootstrap")
        continue
        
        
winsound.Beep(freq, duration)

#%%%%
##SAVE DATA
os.chdir(r"C:\Users\chris\Dropbox (Partners HealthCare)\carTproject\LTR_Results")
df_coef.to_excel("Bootstrap_Coef_12142021.xlsx")
df_cv_te=pd.DataFrame(cv_te_score_tot)
df_cv_te.to_excel("Bootstrap_CV_te_12142021.xlsx")
df_cv_tr=pd.DataFrame(cv_tr_score_tot)
df_cv_tr.to_excel("Bootstrap_CV_tr_12142021.xlsx")
df_ytes=pd.DataFrame(ytes_tot)
df_ytes.to_excel("Bootstrap_ytes_12142021.xlsx")
df_ytes.to_excel("Bootstrap_ytes_12142021.xlsx")
df_yptes=pd.DataFrame(yptes_tot)
df_yptes.to_excel("Bootstrap_yptes_11142021.xlsx")
df_yptes_z=pd.DataFrame(yptes_z_tot)
df_yptes_z.to_excel("Bootstrap_yptes_z_12142021.xlsx")
df_mse=pd.DataFrame(mse_tot)
df_mse.to_excel("Bootstrap_mse_12142021.xlsx")
   
## compute the upper and lower bound for the confidence interval of each coef
#alpha = 0.95
#p = ((1.0-alpha)/2.0) * 100
#print(p)
#lower = np.percentile(coefs, p, axis=0)
#p = (alpha+((1.0-alpha)/2.0)) * 100
#print(p)
#upper = np.percentile(coefs, p, axis=0)
#%%

df_coef=pd.read_excel("Bootstrap_Coef_11242021.xlsx")
##Analyze top 20 features
num_feats=20
coef_counts=df_coef.count(axis=0)

#First sort into top 20 based on inclusion most often in the model, ie 20 most common features
top20=coef_counts.sort_values(ascending=False)[0:num_feats]
top20_mean= df_coef[top20.index.values].mean()
top20_median= df_coef[top20.index.values].median()
coef_CI_top20 = df_coef[top20.index.values].quantile([0.025,0.975])

top20_raw=df_coef[top20.index.values]

#Can either order by median or mean
mean_order = 0
if mean_order == 1:

##Reorder based on mean
#Now reorder dataframe based on the top20_mean absolute value
    top20_mean=top20_mean.sort_values(key=abs,ascending=False)
    top20_ordered=df_coef[top20_mean.sort_values(key=abs,ascending=False).index.values]

#Reorder based on median 
else: 
    top20_median=top20_median.sort_values(key=abs,ascending=False)
    top20_ordered=df_coef[top20_median.sort_values(key=abs,ascending=False).index.values]


#take the # features with highest absolute value of coefficient
num_feats_plot=20
top20_ordered=top20_ordered.iloc[:,0:num_feats_plot]

#then reorder based on actual pos and neg, not absolute value 
if mean_order == 1:
    top20_ordered= top20_ordered[top20_ordered.mean().sort_values(ascending=False).index.values]

##BASED ON MEDIAN
else:
    top20_ordered= top20_ordered[top20_ordered.median().sort_values(ascending=False).index.values]



#%% visualize features


f, ax = plt.subplots(figsize=(7, 6))

###reshape the dataframe to make plot (can use orient ='h' with original, but then issues with palette)
top20_column=top20_ordered.values.reshape(-1,1)
coef_name= top20_ordered.columns.values.reshape(-1,1)
coef_name_column= np.vstack([coef_name]*int(top20_column.shape[0]/num_feats_plot))

#turn into dataframe that is num_feats*num_bootstraps x 2 (for coefficient name and value)
df_coef_h=pd.DataFrame(coef_name_column)
df_coef_h["Feature"]= coef_name_column
df_coef_h["Coefficient Value"]=top20_column
df_coef_h=df_coef_h.iloc[:,1:]

pal = sns.color_palette("coolwarm")

custom_palette = {}
for feat in set(df_coef_h["Feature"]):
    avr = (np.mean(df_coef_h["Coefficient Value"][df_coef_h["Feature"] == feat]))
    if avr < -.5:
        custom_palette[feat] = pal[5]
    elif avr < -.25:
        custom_palette[feat] = pal[4]
    elif avr < 0:
        custom_palette[feat] = pal[3]
    elif avr < .25:
        custom_palette[feat] = pal[2]
    elif avr < .5:
        custom_palette[feat] = pal[1]
    else:
        custom_palette[feat] = pal[0]

sns.boxplot(x="Coefficient Value", y = "Feature", data=df_coef_h,width=.6, palette=custom_palette, showfliers=False,showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":"5"})

#%%

mean_cv_te=np.mean(df_cv_te,axis=1)
np.argmax(mean_cv_te)

f, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(x=df_ytes.values[np.argmax(mean_cv_te)], y=df_yptes.values[np.argmax(mean_cv_te)],palette='Blues',showfliers=False)
sns.stripplot(x=df_ytes.values[np.argmax(mean_cv_te)], y=df_yptes.values[np.argmax(mean_cv_te)],color='gray')

ax.set(xlabel='ICANS', ylabel='E-ICANS')

median_cv_te=np.median(df_cv_te)

ax =sns.histplot(df_ytes.values[0],discrete=True)
ax.set(xlabel='ICANS')

