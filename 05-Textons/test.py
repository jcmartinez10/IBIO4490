# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 07:19:49 2019

@author: luisa
"""
import sys
import numpy as np
import pickle
from skimage import color
from skimage.transform import resize

#Classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Add filter library path
sys.path.append('python')

#Download and open the database 
from cifar10 import load_cifar10
Train = load_cifar10(meta='cifar-10-batches-py', mode=5)
Test = load_cifar10(meta='cifar-10-batches-py',mode='test')

TrainImages=Train["data"]
TrainLabels=Train["labels"]
TestImages=Test["data"]
TestLabels=Test["labels"]

#Transform the images to grayscale
Train = []
for i in range(0,len(TrainImages)):
    img=color.rgb2gray(resize(np.asanyarray(TrainImages[i]), (32, 32)))
    Train.append(img)
    
Test = []
for i in range(0,len(TestImages)):
    img=color.rgb2gray(resize(TestImages[i], (32, 32)))
    Test.append(img)
print('Data loaded')
    

#Create the filters
from fbCreate import fbCreate
fb = fbCreate(support=2, startSigma=0.6) 

#Open Dictionary
infile=open('Diccion300','rb')
Dict=pickle.load(infile)
infile.close()

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

# AssignTextons
from assignTextons import assignTextons
from fbRun import fbRun

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

mapTest=[]
for img in Test:
    mymap=assignTextons(fbRun(fb,img),Dict.transpose())
    mapTest.append(mymap)
print('Test Images textones assign')

histTest=[]   
for map in mapTest:
    histTest.append(histc(map.flatten(), np.arange(300)))
print('Test Images histograms created')

# RandomForest
RF = RandomForestClassifier(n_estimators=600)
RF.fit(histTrain,TrainLabels)
testresultRf=RF.predict(histTest)
AcaTestRF= accuracy_score(TestLabels,testresultRf)
CTestRF=confusion_matrix(TestLabels,testresultRf)/ confusion_matrix(TestLabels,testresultRf).sum(axis=0)

print('Test Results with Random Forest')
print(AcaTestRF)
print(CTestRF)


