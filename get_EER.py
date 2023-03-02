# -*- coding: utf-8 -*-
"""

Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan


Python compute equal error rate (eer) for binary classification
:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

#def EER(path, labels, predicted_labels, positive_label=1):

def EER(path, true_labels, predicted_labels):

    # all fpr, tpr, frr, threshold are lists (in the format of np.array)
    ####When running on my laptop ##python version 3.7 works fine with 3 arguments but when run on python 3.8 throws error for taking only two arguments

    #far, tpr, threshold = roc_curve(true_labels, predicted_labels, positive_label)  

    far, tpr, threshold = roc_curve(true_labels, predicted_labels)
    #print("fpr:", fpr, "tpr:", tpr,"threshold:", threshold)

    #print("False Acceptance Rate is:", far[1]*100)

    ##Plot the FAR
    plt.plot(threshold, far, 'r')
    plt.xlabel('Threshold values')
    plt.ylabel('False Acceptance Rate')
    plt.savefig(path + 'False Acceptance Rate.png')
    #plt.show()
    plt.clf()
    
    ##Plot the FRR
    frr = 1 - tpr

    #print("False Rejection Rate is:", frr[1]*100)

    plt.plot(threshold, frr, 'g')
    plt.xlabel('Threshold values')
    plt.ylabel('False Rejection Rate')
    plt.savefig( path + 'False Rejection Rate.png')
    #plt.show()
    plt.clf()

    # Finding the threshold at which threshold of frr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((frr - far)))]

    #print("eer_threshold:", eer_threshold)

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = far[np.nanargmin(np.absolute((frr - far)))]
    eer_2 = frr[np.nanargmin(np.absolute((frr - far)))]

    #print("eer_1:", eer_1, "eer_2:",eer_2)
    # return the mean of eer from frr and from far
    eer = (eer_1 + eer_2) / 2

    ##Plotting the FAR and FRR in same plot to find the EER
    fig, ax = plt.subplots()
    ax.plot(threshold, far, 'r', label='False Acceptance Rate (FAR)')
    ax.plot(threshold, frr, 'g', label='False Rejection Rate (FRR)')
    plt.xlabel('Threshold')
    plt.ylabel('Rates')

    #plt.plot(1, eer,'ro', label='Equal Error Rate (EER)') 
    legend = ax.legend(loc='upper center', shadow = True) #, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('white')
    plt.savefig( path +'EER.png')
    #plt.show()
    plt.clf()
    
    #print("Equal Error Rate is:", eer*100)

    return far, frr, eer

'''
# sample usage
labels = [1,1,0,0,0]
predicted_labels = [1,1,0,1,0]
eer = compute_eer(labels, predicted_labels)
print(eer)
'''
