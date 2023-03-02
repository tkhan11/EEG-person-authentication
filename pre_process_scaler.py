# -*- coding: utf-8 -*-
"""
Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan
"""

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
import numpy as np
from sklearn import preprocessing
import pandas as pd

Columns = ['Unnamed: 0', 'Unnamed: 0.1', 'PG1', 'FP1', 'FP2', 'PG2', 'F7', 'F3',
       'FZ', 'F4', 'F8', 'A1', 'T3', 'C3', 'CZ', 'C4', 'T4', 'A2', 'T5', 'P3',
       'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5',
       'EX1', 'EX2', 'EX3', 'Label']

Channels = ['FP1', 'FP2','F7', 'F3','FZ', 'F4', 'F8','T3', 'C3', 'CZ', 'C4', 'T4', 'T5', 'P3',
       'PZ', 'P4', 'T6', 'O1', 'OZ', 'O2'] # Label

def preProcess(Training_data, Testing_data):
    np.random.seed(122)

    Training_Data = Training_data.sample(frac = 1)
    Training_Data = Training_Data.drop(['Unnamed: 0','Unnamed: 0.1', 'PG1','PG2',
        'A1', 'A2','BP1', 'BP2', 'BP3', 'BP4', 'BP5','EX1', 'EX2', 'EX3'], axis=1)

   # print(Training_Data.shape)
    Training_Data_features = Training_Data[[x for x in Training_Data.columns if x not in ["Label"]]]   # Data for training

    
    # Combine values of Column1 and Column2 into an array and save in Column3
    Training_Data_features['features'] = list(zip(Training_Data_features['FP1'], Training_Data_features['FP2'],Training_Data_features['F7'],
                              Training_Data_features['F3'], Training_Data_features['FZ'], Training_Data_features['F4'], Training_Data_features['F8'],
                              Training_Data_features['P3'], Training_Data_features['PZ'], Training_Data_features['P4']))

    scaler = preprocessing.MinMaxScaler()
    training_features_list = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in Training_Data_features['features']]
    training_labels_list = [i for i in Training_Data["Label"]]

    print("Length of training features list", len(training_features_list))
    print("Length of training labels list", len(training_labels_list))

    Test_Data = Testing_data.sample(frac = 1)
    Test_Data = Test_Data.drop(['Unnamed: 0','Unnamed: 0.1', 'PG1','PG2',
        'A1', 'A2','BP1', 'BP2', 'BP3', 'BP4', 'BP5','EX1', 'EX2', 'EX3'], axis=1)

    Test_Data_features = Test_Data[[x for x in Test_Data.columns if x not in ["Label"]]]   # Data for testing

    # Combine values of Column1 and Column2 into an array and save in Column3
    Test_Data_features['features'] = list(zip(Test_Data_features['FP1'], Test_Data_features['FP2'],Test_Data_features['F7'],
                              Test_Data_features['F3'], Test_Data_features['FZ'], Test_Data_features['F4'], Test_Data_features['F8'],
                              Test_Data_features['P3'], Test_Data_features['PZ'], Test_Data_features['P4']))

    testing_features_list = [scaler.fit_transform(np.asarray(i).reshape(-1, 1)) for i in Test_Data_features['features']]
    print("Length of testing features list", len(testing_features_list))
    
    testing_labels_list = [i for i in Testing_data["Label"]]
    print("Length of testing labels list", len(testing_labels_list))
  
    test_Y_for_pred = Testing_data['Label']

    seq_length = 512

    drop_feat_values = len(training_features_list) % seq_length
    #print(drop_feat_values)
    training_features_list = training_features_list[:-drop_feat_values]

    drop_labels_values = len(training_labels_list) % seq_length
    #print(drop_labels_values)
    training_labels_list = training_labels_list[:-drop_labels_values] 

    print("Length of training features list after dropping:", len(training_features_list))
    print("Length of training labels list after dropping:", len(training_labels_list))
    
    x_train = np.asarray(training_features_list).astype(np.float32).reshape(-1, 512, 1)
    y_train = np.asarray(training_labels_list).astype(np.float32).reshape(-1,512, 1)
    y_train = keras.utils.to_categorical(y_train)

    
    seq_length = 512

    drop_feat_values = len(testing_features_list) % seq_length
    testing_features_list = testing_features_list[:-drop_feat_values]

    drop_labels_values = len(testing_labels_list) % seq_length
    testing_labels_list = testing_labels_list[:-drop_labels_values]

    print("Length of testing features list after dropping:", len(testing_features_list))
    print("Length of testing labels list after dropping:", len(testing_labels_list))
    
    x_test = np.asarray(testing_features_list).astype(np.float32).reshape(-1, 512, 1)
    y_test = np.asarray(testing_labels_list).astype(np.float32).reshape(-1,512, 1)
    y_test = keras.utils.to_categorical(y_test)

    # Split the train and the validation set for the fitting
    #train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size = 0.15, random_state=2 , stratify = train_Y)
    
    return x_train, y_train, x_test, y_test, test_Y_for_pred

   

