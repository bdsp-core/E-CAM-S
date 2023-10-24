# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 16:40:39 2021

@author: chris
"""
from math import floor
import hdf5storage
import numpy as np
#import cupy as cp
import glob
from mne.filter import filter_data, notch_filter
import pandas as pd

#from joblib import Parallel, delayed
#from scipy.signal import detrend
from scipy.io import savemat
import mne
mne.set_log_level(verbose='WARNING')
from mne.filter import resample

#from peakdetect import peakdetect
#import time
#import random
#from collections import OrderedDict
#from collections import Counter
import os
from datetime import datetime, date
from datetime import timedelta

def getFolderFiles(file_dir, file_type="*.mat"):
    """
    

    Parameters
    ----------
    file_dir : Directory of the files of interest
    file_type : File extension, entered as a string in format "*.ext" The default is "*.mat".

    Returns
    -------
    total_file_list : list of all files with that extension in the folder
    T : total # of files

    """
    os.chdir(file_dir)
#total_file_list = sorted(os.listdir(file_dir))
    total_file_list=glob.glob("*.mat")

    T = len(total_file_list)
    print ("Files: ", T, '\n')
    
    return total_file_list, T

def EEGseg_artifact(X2, win_length, div,amplitude_thres):

    N = int(X2.shape[1]/(div*200))
    chans=np.shape(X2)[0]
    X3 = np.zeros((N-div,chans,200*win_length))

    for n in range(N-div):

        start_sn = int(n*200*div)

        end_sn = start_sn + 200*win_length

        x = X2[:,start_sn:end_sn]

        X3[n,:,:] = x
    
    seg_mask_explanation = [True,False,False,False]  #[1,np.nan,np.nan]
    seg_mask_explanationlab = ['normal','NaN','LargeAmplitude','Std']

    start_ids = np.arange(0, X2.shape[1]-win_length*200, div*200)
    start_ids=start_ids[0: X3.shape[0]]

    if len(start_ids) <= 0:
        raise ValueError('No EEG segments')

    seg_masks = [seg_mask_explanation[0]]*len(start_ids)#defaults to labeling as normal
    seg_maskslab = [seg_mask_explanationlab[0]]*len(start_ids)
  
## find nan in signal

    nan2d = np.any(np.isnan(X3), axis=2)
    nan1d = np.where(np.any(nan2d, axis=1))[0]
    for i in nan1d:
        seg_masks[i] = seg_mask_explanation[1]
        seg_maskslab[i] = seg_mask_explanationlab[1]
    
    # for i in nan1d:
    #     seg_masks[i] = '%s_%s'%(seg_mask_explanation[1], np.where(nan2d[i])[0])

    del nan1d, nan2d, start_ids

    
    amplitude_large2d = (np.max(X3,axis=2)-np.min(X3,axis=2))>2*amplitude_thres 
    amplitude_large1d = np.where(np.any(amplitude_large2d, axis=1))[0]
    for i in amplitude_large1d:
               # seg_masks[i] = '%s_%s'%(seg_mask_explanation[2], np.where(amplitude_large2d[i])[0])
        seg_masks[i] = seg_mask_explanation[2]
        seg_maskslab[i] = seg_mask_explanationlab[2]
    del amplitude_large1d, amplitude_large2d     
    
#look for std <1
    stdlow=np.where(np.std(X3,axis=2)<1)[0]
    for i in stdlow:
        seg_masks[i]=seg_mask_explanation[3]
        seg_maskslab[i] = seg_mask_explanationlab[3]
        
    return seg_masks, seg_maskslab, X3

 
def EEGdatachunk(X3, seg_masks, funkyfile,  perc_target = .8,overlap=2,chunk_time = 60, Fs=200, win_length=10,  n_jobs=-1, verbose='ERROR',to_remove_mean=False):       
        def hourPercent(seg_masks,chunk_time, win_length, div,perc_target):
            #chunk_time : in minutes

            goal=60/chunk_time
            

            chunks= np.arange(0,int(len(seg_masks)-ceil(60/div)),step=ceil(60/div))
            if len(chunks)==0:
                print("No good chunks")
                finalidx=np.nan 
            else:
                def percnorm(x):
                    return np.nansum(seg_masks[x:int(x+(chunk_time*60/div)-1)])/int(chunk_time*60/div)    
                
  
              
                percnorm_vec = np.vectorize(percnorm)
    
                perc_chunks = percnorm_vec(chunks)
    
                goodchunks= np.where(perc_chunks>=perc_target)[0]
                
    
                if np.shape(goodchunks)[0]*div*ceil(60/div)/(60*60) < goal:
                    finalchunks=goodchunks
                elif np.shape(goodchunks)[0]%60 != 0:
                    count = 0
                    
                    counter=int(10*(1-((np.shape(goodchunks)[0]/goal)-floor(np.shape(goodchunks)[0]/goal))))
                    
    
                    for i in np.arange(0,np.shape(goodchunks)[0],step=floor(np.shape(goodchunks)[0]/60)):
                        if counter== 0:
                            if i ==0:
                                idx=np.array(int(i))
                            else:
                                idx=np.append(idx,int(i))
                        elif count <counter:
                            if i ==0:
                                idx=np.array(int(i))
                            else:
                                idx=np.append(idx,int(i))
                            count = count+1
                        else:
                            count = 0
    
                    finalchunks=goodchunks[idx[0:60]]
                else:
                    finalchunks=goodchunks[0::int(np.shape(goodchunks)[0]/goal)]
    
    
                if len(goodchunks)==0:
                    print("No good chunks")
                    finalidx=np.nan 
                else:
                    for i in chunks[finalchunks]:
                        if i ==chunks[finalchunks][0]:
                            finalidx=np.arange(i,i+ceil(60/div),step=int(1))
                        else:
                            finalidx=np.append(finalidx, np.arange(i,i+ceil(60/div),step=int(1)))
            
            return finalidx  #index of EEG segment closest to target time, with total chunk time over threshold perc_target
        
        finalidx=hourPercent(seg_masks, chunk_time, win_length, div, perc_target)
        if np.any(finalidx !=finalidx):
             funkyfile.append(file_name)
             X4=np.nan
           #  start_X4=np.nan 
             segmasks_X4=np.nan
        else:
#            start_X4=np.arange(start, end_time,step=timedelta(seconds=div)).astype(datetime)[finalidx[0]]
            if len(finalidx) ==0:
                print("No viable EEG, skipping file")
                X4=np.nan 
             #   start_X4=np.nan
                segmasks_X4=np.nan
            else:
                X4=X3[finalidx,:,:]
                print("X4 Shape is: " + str(np.shape(X4)))
                seg_masks=np.array(seg_masks)
                segmasks_X4=seg_masks[finalidx]
            
                del X3, seg_masks

        return X4, segmasks_X4, funkyfile        #start_X4, segmasks_X4,
        

#%%
file_dir = r"D:\EEGs\DeliriumEEGs\\"
outdir="D:\EEGs\DeliriumEEGSegments"

target=9 #time in morning
perc_target=.8 #percent needed to be without artifact
chunk_time = 60
Fs=200
win_length=10
notchfreq=60
bandpassfreq=[0.5,20]
amplitude_thres=500
n_jobs=-1
verbose='ERROR'
total_file_list, T = getFolderFiles(file_dir)
overlap = 2
funkyfile=[] 
complete={"ID":[],"Date":[]}
complete=pd.DataFrame(complete)

for t in range(0,T):
     if (t ==1) or (t ==2) or (t==3) or (t==4): #TO DO check these files 
        continue
     else:
        print ("***************************************")
        print ("t: ", t)
        print ("")
         
         
        file_name = total_file_list[t]
         
        print ("file_name: ", file_name)
        print ("")
         
         
        path1 = str(file_dir + file_name)
    
        mat = hdf5storage.loadmat(path1)
        if len(mat) >1:
            dur=mat['data'].shape[1] 
    
            Fs= mat['Fs'][0][0]
    
            print("Duration is " +str(dur/(Fs*60*60)) + "hours.")
         
            X=mat['data']
         
            if Fs != 200:
             #resample to 200 Hz
                 factor= Fs/200
                 X=resample(X,down=factor, n_jobs=1)
    
    
     
        
            print ("X.shape: ", X.shape)
            print ("applying montage")
            print("***************************************")
            print ("")    
            #BIPOLAR MONTAGE
            X2 = X[[0,4,5,6, 11,15,16,17, 0,1,2,3, 11,12,13,14, 9,10]] - X[[4,5,6,7, 15,16,17,18, 1,2,3,7, 12,13,14,18, 10,11]]
              #  fz_cz=EEG[9]-EEG[10]
               # cz_pz=EEG[10]-EEG[11]
            # =============================================================================
            print ("X2.shape: ", X2.shape)
            # 
            # 
            print ("filtering")
            print("***************************************")
            print ("")
        
        
            # =============================================================================
            
            #FILTER
            X2 = notch_filter(X2, 200, notchfreq, n_jobs=n_jobs, verbose='ERROR')
            
            X2 = filter_data(X2, 200, bandpassfreq[0], bandpassfreq[1], n_jobs=n_jobs, verbose='ERROR') 
            
            #LABEL FOR ARTIFACT
           
            #just frontal leads
            X2=X2[[0,4,8,12],:]  #12 isFp1-F7
            #X2=np.vstack((X2,X[19])) #adds back ecg channel after filtering
            chans=np.shape(X2)[0]
            div=win_length-overlap
            seg_masks, seg_maskslab,X3=EEGseg_artifact(X2,win_length,div,amplitude_thres)
                
    
            print("***************************************")
            print ("")
            print("Removing Artifact")

            if np.shape(X3)[0]*div/(60*60) > 1.5: #if EEG is longer than an hour
                 X3, seg_masks, funkyfile=EEGdatachunk(X3,seg_masks,funkyfile)
            X5=X3[seg_masks, :, :]
            
            try:
                start= datetime.strptime(mat['start_time'][0],"%m-%d-%Y %H:%M:%S")
            except:
                start= datetime.strptime(mat['startTime'][0][0],"%d-%b-%Y %H:%M:%S")
            startst_filename = start.strftime('%d-%b-%Y %Hh%Mm%Ss')
            startstr = start.strftime('%d-%b-%Y %H:%M:%S') 
            sid=file_name.split("_")[3][-3:]
            try:
                sid=int(sid)
            except:
                sid=file_name.split("_")[2][-3:]
            
            mdic = dict()
            mdic['SourceFile']=file_name
            mdic['Fs'] = Fs
            mdic['Start'] = startstr
            mdic['Data'] = X5[2:,:,:]
            mdic['ICANS'] = 0
            mdic['sid'] = sid
            
            print("Export file is: ", np.shape(X5))
            print("Duration is ", np.shape(X5)[0]*win_length/(60), " minutes.")
    
            print("Exporting .mat file")
            print("***************************************")
            print ("")
                  #   keeper_files.append(file_name) 
            savemat(os.path.join(outdir, "SID(" + str(int(sid)) + ')_' + file_name[:15] + "_Segment(" + startst_filename + ").mat"), mdic)
            df={"ID":[sid],"Date":[start.date()]}
            df=pd.DataFrame(df)
            complete=complete.append(df)
                    # del X4, segmasks_X4, segswa, startstr, startst_filename, mdic
    
        
     