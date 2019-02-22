#!/usr/bin/python

#Import the modules
import numpy as np
import os
import cv2
import urllib
import zipfile
import matplotlib.pyplot as plt

# Download the images
url = 'https://www.dropbox.com/s/d51zy4yuo1zxcn0/imgs.zip?dl=1'
u = urllib.request.urlopen(url)
data = u.read()
u.close()
with open(os.getcwd() + '/' + 'imgs.zip', "wb") as f :
    f.write(data)
f.close()

# Unzip the images
zip_ref = zipfile.ZipFile(os.getcwd() + '/' + 'imgs.zip', 'r')
zip_ref.extractall(os.getcwd())
zip_ref.close()

img1=cv2.imread('./luisa.png')
img2=cv2.imread('./juancamilo.png')

img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

Dim = 85

img1Gauss = cv2.GaussianBlur(img1, (Dim, Dim), 20)
img2Gauss = cv2.GaussianBlur(img2, (Dim, Dim), 25)
img2dif = cv2.absdiff(img2, img2Gauss)
hybrid = cv2.add(img1Gauss, img2dif)

plt.imshow(np.uint8(hybrid))