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

# Read the images
img1=cv2.imread('./luisa.png')
img2=cv2.imread('./juancamilo.png')

# Puts the images in the rigth format to visualize them
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


d = 85 # Declares the Kernel size

# Create the hybrid image
G1 = cv2.GaussianBlur(img1, (d, d), 20)
G2 = cv2.GaussianBlur(img2, (d, d), 25)
dif = cv2.absdiff(img2, G2)
hybrid = cv2.add(G1, dif)

plt.figure(1)
plt.axis('off')
plt.imshow(np.uint8(hybrid)) # Shows Hybrid Image

# Creates the Gaussian Pyramid of n levels 
rows,cols,_ = hybrid.shape
Pyr = [hybrid.copy()]
for i in range(1,6):
    pyr = cv2.pyrDown(Pyr[i-1])
    Pyr.append(pyr)

# Shows the Gaussian Pyramid
n = 6   # Number of Pyramid levels
comp = np.ones((rows, int((((2**(n)-1)*cols)/(2**(n-1))+2)), 3), dtype=np.uint8)*255
comp[:rows, :cols, :] = Pyr[0]
for p in Pyr[1:]:
    n_rows, n_cols,_ = p.shape
    comp[rows-n_rows:rows, cols:cols + n_cols] = p
    cols=cols + n_cols

plt.figure(2)
plt.axis('off')
plt.imshow(comp)
