#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:21:56 2019

@author: lfborbon
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from skimage import color
from skimage.transform import resize

#Classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.append('python')

if not os.path.isdir(os.path.join(os.getcwd(),'cifar-10-batches-py')):
    os.system('wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    os.system('tar xzvf cifar-10-python.tar.gz')

from cifar10 import load_cifar10
Train = load_cifar10(meta='cifar-10-batches-py', mode=5)

TrainImages=Train["data"]
TrainLabels=Train["labels"]

Train = []
for i in range(0,len(TrainImages)):
    img=color.rgb2gray(resize(np.asanyarray(TrainImages[i]), (32, 32)))
    Train.append(img)
    
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) 

print('Data loaded')

#Set number of clusters
k =  300

TrainArray=np.array([]).reshape(32,0)
for i in range(1000):
    a=color.rgb2gray(TrainImages[i,:,:,:])
    if len(TrainArray)==0:
        TrainArray=a
    else:
        TrainArray=np.concatenate((TrainArray,a),axis=1)
print(TrainArray.shape)

from fbRun import fbRun
filterResponses = fbRun(fb,TrainArray)

from computeTextons import computeTextons
map, Dict = computeTextons(filterResponses, k)

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

# AssignTextons
from assignTextons import assignTextons

mapTrain=[]
#Calculate texton representation with current texton dictionary
for img in Train:
    mymap=assignTextons(fbRun(fb,img),Dict.transpose())
    mapTrain.append(mymap)
print('Train Images textones assign')

histTrain=[]   
for map in mapTrain:
    histTrain.append(histc(map.flatten(), np.arange(300)))
print('Train Images histograms created')

# RandomForest
RF = RandomForestClassifier(n_estimators=600)
RF.fit(histTrain,TrainLabels)
trainresultRf=RF.predict(histTrain)
AcaTestRF= accuracy_score(TrainLabels,trainresultRf)
CTestRF=confusion_matrix(TrainLabels,trainresultRf)/ confusion_matrix(TrainLabels,trainresultRf).sum(axis=0)

print('Test Results with Random Forest')
print(AcaTestRF)
print(CTestRF)
