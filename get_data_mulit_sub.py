# -*- coding: utf-8 -*-
"""
Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan
"""

import pandas as pd

'''
Index(['time', 'PG1', 'FP1', 'FP2', 'PG2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'A1',
       'T3', 'C3', 'CZ', 'C4', 'T4', 'A2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1',
       'OZ', 'O2', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'EX1', 'EX2', 'EX3'],
      dtype='object')
'''


Channels = ['FP1', 'FP2','F7', 'F3', 'FZ', 'F4', 'F8', 'T3','C3','CZ',
            'C4', 'T4', 'T5', 'P3', 'PZ','P4', 'T6', 'O1','OZ', 'O2']


#print(len(Channels))

gen_training_protocols = ['_reo','_reo']
gen_testing_protocols = ['_reo']

forg_training_protocols = ['_reo','_reo']  
forg_testing_protocols = ['_reo']         #Same Protocol


def getEEGData(path, sub):
    Training_data = pd.DataFrame()
    Testing_data = pd.DataFrame()
    
    subjects = ['S1','S2', 'S3', 'S4', 'S5']
    sessions_training = ['S1','S2']
    sessions_testing = ['S3']

    sub_popped = subjects.pop(sub) # list index starts with 0

    print("Subject Popped is:", sub_popped)
    print("Remaining subjects are:", subjects)
    
    
    #First generating genuine training and testing data for one popped subject
    for training_proto in gen_training_protocols:
        for session in sessions_training:
            #path="../Resting_state_dataset/S1/S1S2/S1S2_reo.csv"
            csvfile = pd.read_csv( path+ sub_popped + "/"+ sub_popped + session + "/" + sub_popped + session + training_proto +".csv")       #genuine samples training data
            csvfile.drop(["time"], axis="columns", inplace=True)
            subject_label = sub_popped.replace(sub_popped,'1')
            csvfile['Label'] = int(subject_label)
            Training_data = pd.concat([Training_data, csvfile], ignore_index=True)
        
    for testing_proto in gen_testing_protocols:
        for session in sessions_testing:
            ##Testing data for genuine user
            csvfile = pd.read_csv( path+ sub_popped + "/"+ sub_popped + session + "/" + sub_popped + session + training_proto +".csv")       #genuine samples testing data
            csvfile.drop(["time"], axis="columns", inplace=True)
            subject_label = sub_popped.replace(sub_popped,'1')
            csvfile['Label'] = int(subject_label)
            Testing_data = pd.concat([Testing_data, csvfile], ignore_index=True)

    #Now generating forged training and testing data for other 4 subjects
    for training_proto in forg_training_protocols:
        for subject in subjects:
            for session in sessions_training:
                csvfile = pd.read_csv( path+ subject + "/"+ subject + session + "/" + subject + session + training_proto +".csv")       #forged samples training data
                csvfile.drop(["time"], axis="columns", inplace=True)
                csvfile = csvfile[3750:7500] ##Taking 15s data starting from 15s to 30s from every other 4 subjects for 2 sessions to make it of length 2min as it is equal to training data of genuine subject
                subject_label = subject.replace(subject,'0')
                csvfile['Label'] = int(subject_label)
                Training_data = pd.concat([Training_data, csvfile], ignore_index=True)
        
    for testing_proto in forg_testing_protocols:
        for subject in subjects:
            for session in sessions_testing:
                csvfile = pd.read_csv( path+ subject + "/"+ subject + session + "/" + subject + session + training_proto +".csv")       #forged samples testing data

                csvfile.drop(["time"], axis="columns", inplace=True)
                csvfile = csvfile[3750:7500] ##Taking 15s data starting from 15s to 30s from every other 4 subjects to make it of length 1min as it is equal to testing data of genuine subject
                subject_label = subject.replace(subject,'0')
                csvfile['Label'] = int(subject_label)
                Testing_data = pd.concat([Testing_data, csvfile], ignore_index=True)
    
    return Training_data, Testing_data
