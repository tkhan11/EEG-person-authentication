# -*- coding: utf-8 -*-
"""
Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan
"""

import pandas as pd
import time

from get_data_mulit_sub import getEEGData

from pre_process_scaler import preProcess

from GRU_model_train_PA import modelTrainPA
from model_eval import evaluateModel

from plot_CM import plot_confusion_matrix
from get_EER import EER


import warnings
warnings.filterwarnings('ignore')

seconds= time.time()
time_start = time.ctime(seconds)   
print("start time:", time_start,"\n")    

#Setting some variables
sub_labels = ['Forged','Genuine']

##When Generating the Train and test Data uncomment below line of code
path = "../Resting_state_dataset/"

subjects_ids = ['S1']# 'S1', 'S2', 'S3', 'S4', 'S5']

epochs = 5
batch_size = 1025

False_Acceptance_rate = []
False_Rejection_rate = []
Equal_Error_rate = []

for sub in subjects_ids:
    sub_index = subjects_ids.index(sub)

    sub_index = sub_index 

    ##Getting the train and test data
    ##Loading data time
    load_start = time.time()
    
    Training_data, Testing_data = getEEGData(path, sub_index)
    load_stop = time.time()

    time_to_load_data = load_stop - load_start
    print("\n\nTime Taken(in seconds) to load data:", time_to_load_data )

    
    print("\nShape of Training and Test data for subject "+ sub + " is:" , Training_data.shape, Testing_data.shape)
    print("\nValue count of forged (0) and genuine (1) samples in Training data for subject " + sub + " is:\n" , Training_data.Label.value_counts())
    print("\nValue count of forged (0) and genuine (1) samples in Test data for subject " + sub + " is:\n" , Testing_data.Label.value_counts())

    ##Pre-processing the EEG data to get to the desired shape
    X_train, Y_train, X_test, Y_test, test_Y_for_pred = preProcess(Training_data, Testing_data)

    print(f"Length of X_train : {len(X_train)}\nLength of X_test : {len(X_test)}\nLength of Y_train : {len(Y_train)}\nLength of Y_test : {len(Y_test)}")
    
    '''
    ###Train the deep learning model on training and Validation data
    modelTrainPA(sub, epochs, batch_size, train_X, train_Y, val_X, val_Y)
    
    
    #print("Waiting for 5 sec.")
    time.sleep(5) # Delay for 5 seconds.
    
    dir_path = './subjects_saved_models/model/' + sub + "/"
    saved_model_path= './subjects_saved_models/model/' + sub + '/'+ sub + "GRU_PA_sub_1_25epochs_128batch.h5"
        
    print("\n\nModel Evaluation Starts Here!!!!!!!!!!!!!!!!\n")
    #Evaluate the performance of trained model on Test data This function returns confusion_matrix and predicted subject probability values
    confusion_matrix, predicted_subject_prob, predicted_subject = evaluateModel(saved_model_path, batch_size, test_X, test_Y, test_Y_for_pred)  

    ###Plot the confusion matrix
    plot_confusion_matrix(dir_path, confusion_matrix, sub_labels)

    ###Finding the EER
    FAR, FRR, EER_rate = EER(dir_path, test_Y_for_pred, predicted_subject)

    False_Acceptance_rate.append(FAR[1])            ## At threshold 1
    False_Rejection_rate.append(FRR[1])             ## At threshold 1
    Equal_Error_rate.append(EER_rate)            ## At threshold 1
    
    print("\nWaiting for other subject's data to load!!!!!!!!!!!!\n")
    
    result_metrics = pd.DataFrame({'FAR':False_Acceptance_rate,'FRR':False_Rejection_rate, 'EER':Equal_Error_rate}) 
    result_metrics.to_csv(dir_path + '/result_metrics.csv',index=False)

    '''
