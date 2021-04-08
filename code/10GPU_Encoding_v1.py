# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 23:48:38 2019

@author: zhenh
"""

import scipy.io as sio
from scipy import signal
import os
import numpy as np
import keras
from keras.models import Model, Sequential
from keras.layers import Dense,Input, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
import matplotlib.pyplot as plt
from keras.callbacks import History, EarlyStopping
from sklearn import decomposition
from sklearn.svm import SVC
import os
import pandas as pd
import math
import tensorflow as tf
import time

os.environ["CUDA_VISIBLE_DEVICES"]="2"

radi = np.arange(1,64,4)
subjects = ['S00', 'S01','S02','S03','S04','S05','S06','S07','S08','S10','S11','S13','S14','S16','S17','S19','S20','S21','S22','S24'];
unorderflag = 0;
 
for v in range(13,20):
    RANGE_START = v    
    if(RANGE_START > 99):
        unorderflag = 1;
        RANGE_START = RANGE_START - 100     
    
    #for a in range(4,8):
    for a in range(RANGE_START,RANGE_START+1):        
        print('Subject ' + str(a))        
        folds = 5;
        timeMulti = 200; #For neural network, reccomend timeSmooth=1
        methodFlag = 2; #0 for SVM, 1 for NN
        timeSmooth = 5; #Gaussian temporal smoothing of time series must be odd
        
        prereccurent_downsample =40
        radius_downsample =4
        
        temporalSmooth = signal.gaussian(timeSmooth, std=1) 
        temporalSmooth = temporalSmooth/np.sum(temporalSmooth)
        if(1):
        
            f = sio.loadmat('/home/zhenh/Output_wk234_256Hz_CSD_Filter/' + subjects[a] + 'InphasePreCSD.mat')
            Load2_InphasePre = f['Load2']
            Load4_InphasePre = f['Load4']
            Load6_InphasePre = f['Load6']
            print('Inphase')
            f = sio.loadmat('/home/zhenh/Output_wk234_256Hz_CSD_Filter/' + subjects[a] + 'ShamPreCSD.mat')
            Load2_InShamPre = f['Load2']
            Load4_InShamPre = f['Load4']
            Load6_InShamPre = f['Load6']
            print('Sham')
            f = sio.loadmat('/home/zhenh/Output_wk234_256Hz_CSD_Filter/' + subjects[a] + 'AntiphasePreCSD.mat')
            Load2_AntiphasePre = f['Load2']
            Load4_AntiphasePre = f['Load4']
            Load6_AntiphasePre = f['Load6']
            print('Antiphase')
            
            Load2 = np.concatenate((Load2_InphasePre,Load2_InShamPre,Load2_AntiphasePre),axis=2)
            Load4 = np.concatenate((Load4_InphasePre,Load4_InShamPre,Load4_AntiphasePre),axis=2)
            Load6 = np.concatenate((Load6_InphasePre,Load6_InShamPre,Load6_AntiphasePre),axis=2)
            
            sz = np.shape(Load2)
            numLoad2 = sz[2]
            sz = np.shape(Load4)
            numLoad4 = sz[2]
            sz = np.shape(Load6)
            numLoad6 = sz[2]
            
            #Normalize
            Load = np.concatenate((Load2, Load4, Load6),axis=2)
         
            for b in range(sz[0]):
                baselineMean = np.mean(Load[b,0:256,:].flatten())
                baselineSD = np.sqrt(np.var(Load[b,0:256,:].flatten()))
                Load2[b,:,:] = (Load2[b,:,:] - baselineMean)/baselineSD
                Load4[b,:,:] = (Load4[b,:,:] - baselineMean)/baselineSD
                Load6[b,:,:] = (Load6[b,:,:] - baselineMean)/baselineSD
                
            print('Normalize')            
            #Temporal Smooth
            if(timeSmooth > 1):
                flr = np.floor(timeSmooth/2)
                sz = np.shape(Load2)
                Load2_T = np.zeros((sz[0],int(sz[1] - 2*flr),sz[2]))
                sz = np.shape(Load4)
                Load4_T = np.zeros((sz[0],int(sz[1] - 2*flr),sz[2]))
                sz = np.shape(Load6)
                Load6_T = np.zeros((sz[0],int(sz[1] - 2*flr),sz[2]))
                          
                               
                bCount = -1;
                #for b in range(int(flr),int(sz[1]-flr)):
                for b in range(325,int(325+timeMulti*3)): 
                    bCount = bCount + 1;
                    for c in range(sz[0]):
                        for d in range(numLoad2):
                            timx = np.arange(int(b-flr),int(b+flr+1))
                            Load2_T[c,int(bCount),d] = np.dot(np.squeeze(Load2[c,timx,d]),temporalSmooth)
                        for d in range(numLoad4):
                            Load4_T[c,int(bCount),d] = np.dot(np.squeeze(Load4[c,timx,d]),temporalSmooth)
                        for d in range(numLoad6):
                            Load6_T[c,int(bCount),d] = np.dot(np.squeeze(Load6[c,timx,d]),temporalSmooth)
                            
                del Load2
                del Load4
                del Load6
                Load2 = Load2_T
                Load4 = Load4_T
                Load6 = Load6_T
                
                print('Smooth')                           
        
        for q in range(0,len(radi)):
            timeRadius = radi[q]; #For recurrent classifier
            print('Radi ' + str(q))                         
                    
            if(methodFlag == 2):
                #Load = np.concatenate((Load2, Load4, Load6),axis=2)
                Load = np.concatenate((Load2, Load6),axis=2)
                sz = np.shape(Load)
                Label = np.zeros((sz[2],3))
                bCount = -1
                for b in range(numLoad2):
                    bCount = bCount+1
                    Label[bCount,:] = [1,0,0]
        #        for b in range(numLoad4):
        #            bCount = bCount+1
        #            Label[bCount,:] = [0,1,0]
                for b in range(numLoad6):
                    bCount = bCount+1
                    Label[bCount,:] = [0,0,1]
                    
                #indx = np.arange(numLoad2+numLoad4+numLoad6)
                indx = np.arange(numLoad2+numLoad6)
                np.random.shuffle(indx)
                Load = Load[:,:,indx]
                Label = Label[indx,:]                
                            
                #############################################################
                for b in range(0,1):
                    
        #            sectionSize = int((numLoad2+numLoad4+numLoad6)/folds)
        #            foldGuide = np.zeros(int((numLoad2+numLoad4+numLoad6)),) 
        #            for bb in range(int((numLoad2+numLoad4+numLoad6))):
        #                for c in range(folds):
        #                    if bb >= c*sectionSize:
        #                        foldGuide[bb] = foldGuide[bb] + 1
                                
                    sectionSize = int((numLoad2+numLoad6)/folds)
                    foldGuide = np.zeros(int((numLoad2+numLoad6)),) 
                    for bb in range(int((numLoad2+numLoad6))):
                        for c in range(folds):
                            if bb >= c*sectionSize:
                                foldGuide[bb] = foldGuide[bb] + 1
                            
                    #time radius encoding            
                    sz = np.shape(Load)
                    Load_T = np.zeros((sz[0],timeMulti,sz[2],timeRadius))
                    for bb in range(sz[0]):
                        for c in range(sz[2]):
                            for k in range(timeMulti):
                           
                                eCount = -1
                                for e in range(0,int(timeRadius*radius_downsample),radius_downsample):
                                    eCount = eCount + 1
                                    Load_T[bb,k,c,int(eCount)] = Load[bb,int(b+e+k),c] 
                                    
                             
                    #Time Multiplex
                    sz = np.shape(Load_T)
                    Label_T = np.zeros((sz[2]*timeMulti,3))
                    foldGuide_T = np.zeros(sz[2]*timeMulti,)
                    Load_TT = np.zeros((sz[0],1,int(sz[2]*timeMulti),sz[3]))
                    bcCount = -1
                    for bb in range(sz[2]):
                        for c in range(timeMulti):
                            bcCount = bcCount + 1
                            Label_T[int(bcCount),:] = Label[bb,:]
                            foldGuide_T[int(bcCount)] = foldGuide[bb]
                            for d in range(sz[0]):
                                for f in range(sz[3]):
                                    Load_TT[d,0,int(bcCount),f] = Load_T[d,c,bb,f]
                                    
                   
                  
                    del foldGuide
                    foldGuide = foldGuide_T
                    
                    sz = np.shape(Load_TT)
                    numTrain = sz[2]
                    indx = np.arange(numTrain)
                    np.random.shuffle(indx)
                    Load_TT = Load_TT[:,:,indx,:]
                    Label_T = Label_T[indx,:]
                    foldGuide = foldGuide[indx]
                    
                        
                    accurracies = np.zeros((folds,1))
                    t = time.time()
                    for c in range(folds):
                        testingIDX = np.empty(1,)
                        trainingIDX = np.empty(1,)
                        validationIDX = np.empty(1,)
                        for d in range(sz[2]):
                            if(c+1 == int(foldGuide[d])):
                                testingIDX = np.concatenate((testingIDX,[d]),axis=0)
                            
                            elif c < 4 and c+2 == int(foldGuide[d]):
                                validationIDX = np.concatenate((validationIDX,[d]),axis=0)
                            elif c == 4 and c == int(foldGuide[d]):
                                validationIDX = np.concatenate((validationIDX,[d]),axis=0)
                            else:
                                trainingIDX = np.concatenate((trainingIDX,[d]),axis=0)
                           
                        testingIDX = testingIDX[1:]
                        trainingIDX = trainingIDX[1:]
                        validationIDX = validationIDX[1:]
                   
                            
                        testingDATA = np.swapaxes(Load_TT[:,0,testingIDX.astype(int),:],0,1)
                        testingDATA = np.swapaxes(testingDATA,1,2)
                        testingLABEL = Label_T[testingIDX.astype(int)]
                   
                        trainingDATA = np.swapaxes(Load_TT[:,0,trainingIDX.astype(int),:],0,1)
                        trainingDATA = np.swapaxes(trainingDATA,1,2)
                        trainingLABEL = Label_T[trainingIDX.astype(int)]
                        
                        validationDATA = np.swapaxes(Load_TT[:,0,validationIDX.astype(int),:],0,1)
                        validationDATA = np.swapaxes(validationDATA,1,2)
                        validationLABEL = Label_T[validationIDX.astype(int)]
                        
                        shp = np.shape(trainingDATA)
                        inShp = [shp[1],shp[2]]
                        
                        if(unorderflag == 1):
                            #scramble time series
                            scramble = np.arange(timeRadius)
                            np.random.shuffle(scramble)
                            validationDATA = validationDATA[:,scramble,:]                        
                            trainingDATA = trainingDATA[:,scramble,:]                    
                            trainingDATA = trainingDATA[:,scramble,:]   
                        
                        
                        
                        Decoder = Sequential();
                        Decoder.add(LSTM(units=50, return_sequences=False, input_shape = inShp));
                        Decoder.add(Dense(units=3, activation="softmax"))
                        callback = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5,restore_best_weights=True)]
                        Decoder.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']);
                        history = Decoder.fit(trainingDATA, trainingLABEL, epochs=100, verbose=0, validation_data = (validationDATA,validationLABEL), batch_size=32, callbacks=callback)
                        training_loss_curve = history.history['loss']
                        val_loss_curve = history.history['val_loss']
            
                        prediction = Decoder.predict(testingDATA)
                        shp = np.shape(prediction)
                        count = 0
                        for d in range(shp[0]):
                            pred = np.argmax(prediction[d,:])
                            actual = np.argmax(testingLABEL[d,:])
                            if (pred == actual):
                                count = count + 1
                           
                        accurracies[c,0] = count/shp[0]
                         
                    per = ((q+1)*(a+1))/(len(subjects)*len(radi))
                    print("done " + str(per))
                         
                
                    if(unorderflag == 0):
                        decodRadi = np.zeros((len(radi),len(subjects)))
                        decodRadi[q,a] = np.mean(accurracies)
                        decodRadi_POOL = np.load('/home/zhenh/Silent/decodRadi_order_encoding.npy')
                        decodRadi_POOL = decodRadi + decodRadi_POOL;
                        np.save('/home/zhenh/Silent/decodRadi_order_encoding.npy',decodRadi_POOL)
                        elapsed = time.time() - t
                        print(elapsed) 
                        
                    if(unorderflag == 1):
                        decodRadi = np.zeros((len(radi),len(subjects)))
                        decodRadi[q,a] = np.mean(accurracies)
                        decodRadi_POOL = np.load('/home/zhenh/Silent/decodRadi_unorder_encoding.npy')
                        decodRadi_POOL = decodRadi + decodRadi_POOL;
                        np.save('/home/zhenh/Silent/decodRadi_unorder_encoding.npy',decodRadi_POOL)
                        elapsed = time.time() - t
                        print(elapsed)                    
                                