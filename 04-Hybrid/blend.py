# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 17:28:41 2019

@author: luisa
"""

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

# Reads the images
img1=cv2.imread('./luisa.png')
img2=cv2.imread('./juancamilo.png')

img1 = cv2.resize(img1,(int(512),int(512)))
img2 = cv2.resize(img2,(int(512),int(512)))

# Puts them into the rigth format
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

im1 = img1.copy()
Gauss1 = [im1]
for i in range(1,6):
	im1 = cv2.pyrDown(im1)
	Gauss1.append(im1)

# generate Gaussian pyramid for B
im2 = img2.copy()
Gauss2 = [im2]
for i in range(1,6):
	Gauss2 = cv2.pyrDown(im2)
	Gauss2.append(im2)
# generate Laplacian Pyramid for A
lap1 = [Gauss1[5]]
for i in range(5,0,-1):
	GE = cv2.pyrUp(Gauss1[i])
	L = cv2.subtract(Gauss1[i-1],GE)
	lap1.append(L)

# generate Laplacian Pyramid for B
lap2 = [Gauss2[5]]
for i in range(5,0,-1):
	GE = cv2.pyrUp(Gauss2[i])
	L = cv2.subtract(Gauss2[i-1],GE)
	lap2.append(L)

LS = []
for la,lb in zip(lap1,lap2):
	rows,cols,dpt = la.shape
	cols=cols/2
	cols=int(cols)
	ls = np.hstack((la[:,0:cols], lb[:,cols:]))
	LS.append(ls)

# Now reconstruct
ls_ = LS[0]
for i in range(1,6):
	ls_ = cv2.pyrUp(ls_)
	ls_ = cv2.add(ls_,LS[i])

# image with direct connecting each half

plt.imshow(ls_)
plt.show()

cv2.imwrite('Pyramid_blending2.jpg',ls_)

#References
# https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html