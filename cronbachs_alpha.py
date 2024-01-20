# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 11:25:03 2024

@author: WS3

cronbach's  alpha for interrater reliability 
"""

#%% Loading modules
import yasa
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
pip install pingouin
import pingouin as pg

#%% Loading the raw data


#selecting EEG file of session 1 of all subjects
# Parent path
root_dir_eegfile = r"/serverdata/ccshome/adla/NAS/CCS_SleepScorers/Dozee_sleepdata/Dozee Raw data/"

# Checking the files
for path in glob.glob(f'{root_dir_eegfile}/*/*/*_0001.edf', recursive=True):
    print("File Name: ", path.split('\\')[-1])
    print("Path: ", path)

# Files in a list
eegfiles = glob.glob(f'{root_dir_eegfile}/*/*/*')

    
filepath = glob.glob(f'{root_dir_eegfile}/*/*/*_0001.edf')


common = set(eegfiles) & set(filepath)
eegfiles = [i for i in eegfiles if i not in common]

#%% finding the percent of each sleep stage of each sbjct in yasa, krishan and adla scored data


mastersheet = pd.DataFrame()
stage_ratios = pd.DataFrame()


for i in range(len(eegfiles)):
    try:
        
        # Loadin theraw eeg file
        raw = mne.io.read_raw_edf(eegfiles[i] , preload = False, verbose = True)
        
        fname = os.path.basename(eegfiles[i])[:-4]
        
        # Loading the scored hypnograms
        hypno_k = np.loadtxt(f'{fname}_0001_reduced_krishan.csv',dtype = str)
        hypno_a = np.loadtxt(f'{fname}_0001_reduced_scoredbyadla.csv',dtype = str)
        
        # Scoring through yasa 
        sls = yasa.SleepStaging(raw, eeg_name = "EEG CZ-Ref")
        hypno_y = sls.predict()
        
        # Calculating the percent of each sleep stage wrt to total number of sleep stages for k
        k_stage_w = pd.DataFrame(hypno_k).value_counts()['W']*100/ pd.DataFrame(hypno_k).count()
        k_stage_n1 = pd.DataFrame(hypno_k).value_counts()['N1']*100/ pd.DataFrame(hypno_k).count()
        k_stage_n2 = pd.DataFrame(hypno_k).value_counts()['N2']*100/ pd.DataFrame(hypno_k).count()
        k_stage_n3 = pd.DataFrame(hypno_k).value_counts()['N3']*100/ pd.DataFrame(hypno_k).count()
        k_stage_r = pd.DataFrame(hypno_k).value_counts()['R']*100/ pd.DataFrame(hypno_k).count()
    
        
        # Calculating the percent of each sleep stage wrt to total number of sleep stages for a
        a_stage_w = pd.DataFrame(hypno_a).value_counts()['W']*100/ pd.DataFrame(hypno_a).count()
        a_stage_n1 = pd.DataFrame(hypno_a).value_counts()['N1']*100/ pd.DataFrame(hypno_a).count()
        a_stage_n2 = pd.DataFrame(hypno_a).value_counts()['N2']*100/ pd.DataFrame(hypno_a).count()
        a_stage_n3 = pd.DataFrame(hypno_a).value_counts()['N3']*100/ pd.DataFrame(hypno_a).count()
        a_stage_r = pd.DataFrame(hypno_a).value_counts()['R']*100/ pd.DataFrame(hypno_a).count()
        
        # Calculating the percent of each sleep stage wrt to total number of sleep stages for yasa
        y_stage_w = pd.DataFrame(hypno_y).value_counts()['W']*100/ pd.DataFrame(hypno_y).count()
        y_stage_n1 = pd.DataFrame(hypno_y).value_counts()['N1']*100/ pd.DataFrame(hypno_y).count()
        y_stage_n2 = pd.DataFrame(hypno_y).value_counts()['N2']*100/ pd.DataFrame(hypno_y).count()
        y_stage_n3 = pd.DataFrame(hypno_y).value_counts()['N3']*100/ pd.DataFrame(hypno_y).count()
        y_stage_r = pd.DataFrame(hypno_y).value_counts()['R']*100/ pd.DataFrame(hypno_y).count()
    
        # Saving these values into a list
        stage_ratios_k =  [k_stage_w, k_stage_n1, k_stage_n2, k_stage_n3, k_stage_r]
        stage_ratios_a =  [a_stage_w, a_stage_n1, a_stage_n2, a_stage_n3, a_stage_r]
        stage_ratios_y =  [y_stage_w, y_stage_n1, y_stage_n2, y_stage_n3, y_stage_r]
    
        
        # Stage ratios saved into a mastersheet with subject name stagewise
        stage_ratios = pd.concat([pd.DataFrame(stage_ratios_a), pd.DataFrame(stage_ratios_k), pd.DataFrame(stage_ratios_y)], axis = 1)
        stage_ratios['subjname'] = fname
        mastersheet = mastersheet._append(stage_ratios)
        print(i)
    except:
        print("Mission Failed")



mastersheet.columns = ['Adla', 'Krishan', 'Yasa', 'subjname'] 

mastersheet.to_csv(r"/serverdata/ccshome/adla/NAS/Adla/mastersheet_stageratios.csv")

#%% finding the percent of each sleep stage aross sbjcts in krishan and adla scored data
avg_mastrsheet_pre = mastersheet.drop(['Yasa','subjname'], axis = 1)

avg_stge_all = pd.DataFrame()
for i in [0,1,2,3,4]:
    avg_stage = np.mean(avg_mastrsheet_pre.loc[i],axis = 0)
    avg_stge_all = avg_stge_all._append(avg_stage, ignore_index= True)


#%% cronbach alpha 

pip install pingouin
import pingouin as pg

pg.cronbach_alpha(data=avg_stge_all)   # result = 0.982

