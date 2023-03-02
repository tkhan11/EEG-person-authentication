"""
Created on Wed March 1 12:11:34 2023

@author: Tanveer Khan
"""

import matplotlib.pyplot as plt
import time
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, Activation, GRU,Reshape, BatchNormalization
import os

import warnings
warnings.filterwarnings('ignore')

subjects_saved_models_path = './subjects_saved_models/model1_sub_1/'

if os.path.exists(subjects_saved_models_path) == False:
    os.mkdir(subjects_saved_models_path)


def modelTrainPA(sub,epochs,batch_size, train_X, train_Y, val_X, val_Y):
    
    n_timesteps, n_features, n_outputs = train_X.shape[1], train_X.shape[2], train_Y.shape[1]

    model = Sequential()
    
    model.add(GRU(128, return_sequences=True, input_shape= (n_timesteps,n_features)))
    model.add(Dropout(0.3))
    
    model.add(GRU(64, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(Dense(256, activation='relu'))
    model.add(Flatten())

    model.add(Dense(n_outputs, activation = "softmax"))

    '''
    TRY THIS BIDIRECTIONAL GRU MODEL_2
    
    model.add(Bidirectional(GRU(220, return_sequences=True, input_shape= (n_timesteps,n_features))))
    model.add(Dropout(0.2))
    
    model.add(Bidirectional(GRU(220, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(Flatten())

    model.add(Dense(n_outputs, activation = "softmax"))
    '''
    
    # Define the optimizer
    #optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = Adam(learning_rate=0.002)

    # Compile the model
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    # Model Training
    model_history = model.fit(train_X, train_Y, batch_size = batch_size, epochs = epochs,validation_data = (val_X, val_Y))

    print(model.summary())
    
    seconds= time.time()
    time_stop = time.ctime(seconds)   
    print("Stop time:", time_stop,"\n") 

    ###Saving the trained model and Val_loss_acc curve to particular subject dir
    subject_path = os.path.join(subjects_saved_models_path, sub)
    if os.path.exists(subject_path) == False:
            os.mkdir(subject_path)
    
    model.save(os.path.join(subject_path, sub +'GRU_PA_sub_1_' + str(epochs) + 'epochs_' + str(batch_size) + 'batch.h5'))
    
    path = subject_path + "/"
    
    ####FOR PLOTTING
    # Training and validation curves
    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    ax[0].plot(model_history.history['loss'], color='b', label="Training loss")
    ax[0].plot(model_history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(model_history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(model_history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig(path +'GRU_PA_sub_1_' + str(epochs) + 'epochs_' + str(batch_size) + 'batch_val_loss_acc.png')
    #plt.show()
    plt.clf()
                     
