#!/usr/bin/env python
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from skimage import io, color
import matplotlib.pyplot as plt
from skimage.transform import resize


# load dtm image
img = color.rgb2hsv(io.imread('2092.jpg'))
xsize=img.shape[0]
ysize=img.shape[1]
img=resize(img, (int(img.shape[0] / 6), int(img.shape[1] / 6)))

# k-means clustering of the image (population of pixels intensities)
X = img.reshape((-1, 1))
hie = AgglomerativeClustering(n_clusters=2, linkage='average')
hie.fit(X)

# extract means of each cluster & clustered population
X_clustered = hie.labels_

X_clustered.shape = img.shape
X_clustered=resize(img, (xsize, ysize))
print(X_clustered.shape)
print(X_clustered[:,:,1])
X_segmented=X_clustered[:,:,0]+X_clustered[:,:,1]+X_clustered[:,:,2]
#X_clustered=resize(img, (xsize, ysize))
X_clustered=(255*X_clustered).astype(np.uint8)
io.imsave('aha.png',X_clustered)

plt.imshow(X_segmented, cmap=plt.get_cmap('viridis')) #or another colormap that you like https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()
