# -*- coding: utf-8 -*-
"""
Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan
"""
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



def evaluateModel(path,batch_size, test_X, test_Y, test_Y_for_pred):
    model = load_model(path)

    # evaluate model
    #_, accuracy = model.evaluate(test_X, test_Y, batch_size)

    # Predict the values from the Test dataset
    predicted_subject_prob = model.predict(test_X)
    # Convert predictions classes to one hot vectors 
    predicted_subject = np.argmax(predicted_subject_prob,axis = 1) 
    # computing the confusion matrix
    confusion_mtx = confusion_matrix(test_Y_for_pred, predicted_subject) 

    sub_labels = ['Forged','Genuine']
    #Printing Classification Report
    print(classification_report(test_Y_for_pred, predicted_subject, target_names = sub_labels))

    accuracy = accuracy_score(test_Y_for_pred, predicted_subject)
    print('Accuracy: %f' % accuracy)

    return confusion_mtx, predicted_subject_prob, predicted_subject
