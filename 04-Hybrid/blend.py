# -*- coding: utf-8 -*-
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

# Resize the image to a number that is a power of two (2^9)
img1 = cv2.resize(img1,(int(512),int(512)))
img2 = cv2.resize(img2,(int(512),int(512)))

# Puts the images in the rigth format to visualize them
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


# Generate Gaussian pyramid for the images
Gauss1 = [img1.copy()]
Gauss2 = [img2.copy()]
for i in range(1,6):
    im1 = cv2.pyrDown(Gauss1[i-1])
    im2 = cv2.pyrDown(Gauss2[i-1])	
    Gauss1.append(im1)
    Gauss2.append(im2)

# Generate Laplacian Pyramid for the images
lap1 = [Gauss1[5]]
lap2 = [Gauss2[5]]
for i in range(5,0,-1): 
	L1 = cv2.subtract(Gauss1[i-1],cv2.pyrUp(Gauss1[i]))
	L2 = cv2.subtract(Gauss2[i-1],cv2.pyrUp(Gauss2[i]))
	lap1.append(L1)
	lap2.append(L2)

# Concatenate the images
GL = []
for la,lb in zip(lap1,lap2):
	_,cols,_ = la.shape
	GL.append(np.hstack((la[:,0:int(cols/2)], lb[:,int(cols/2):])))

# Reconstruct to create the blended image
blend = GL[0]
for i in range(1,6): 
	blend = cv2.add(cv2.pyrUp(blend),GL[i])

# Shows the final blended image
plt.figure(1)
plt.axis('off')
plt.imshow(blend)

#References
# This code was based on the example found in:
# https://docs.opencv.org/3.1.0/dc/dff/tutorial_py_pyramids.html