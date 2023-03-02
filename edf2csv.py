import pandas as pd
import numpy as np
import os
import time
import mne
from mne import find_events, fit_dipole

'''
seconds= time.time()
time_start = time.ctime(seconds)   #  The time.ctime() function takes seconds passed since epoc
print("start time:", time_start,"\n")    # as an argument and returns a string representing time.

'''

import warnings
warnings.filterwarnings("ignore")

'''
Index(['time', 'PG1', 'FP1', 'FP2', 'PG2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'A1',
       'T3', 'C3', 'CZ', 'C4', 'T4', 'A2', 'T5', 'P3', 'PZ', 'P4', 'T6', 'O1',
       'OZ', 'O2', 'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'EX1', 'EX2', 'EX3'],
     dtype='object')
'''


#Dataset Path= "C:\Users\tanveerlaptop\Desktop\REEDCON paper\Dataset\S1\S1_S1\S1S1BMT.edf"

# Read Data
subject_dir="CSVDataset"

if os.path.exists(subject_dir) == False:
    os.mkdir(subject_dir)


for s in range(1,6):
    
    Subject_path = os.path.join(subject_dir,"S"+str(s))
    
    if os.path.exists(Subject_path) == False:
        os.mkdir(Subject_path)
    
    for t in range(1,4):
        dataset_path = "./Dataset/S"+str(s)+"/S"+ str(s)+"_S"+ str(t) +"/S" + str(s)+"S"+ str(t) + "BMT.edf"
        #print(dataset_path)
        data = mne.io.read_raw_edf(dataset_path, preload=True)
        dataset = data.to_data_frame()

        Subject_id = "S"+ str(s)+"S"+ str(t)
        subject_saved_path = os.path.join(Subject_path, Subject_id)

        if os.path.exists(subject_saved_path) == False:
            os.mkdir(subject_saved_path)

        dataset.to_csv(os.path.join(subject_saved_path, Subject_id + ".csv"))
    
