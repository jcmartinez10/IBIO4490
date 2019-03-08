#!/usr/bin/env python
import numpy as np
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from skimage import io, color
import matplotlib.pyplot as plt


# load dtm image
img = color.rgb2hsv(io.imread('2092.jpg'))

# k-means clustering of the image (population of pixels intensities)
X = img.reshape((-1, 1))

gmm = GaussianMixture(n_components=2)
gmm.fit(X)

# extract means of each cluster & clustered population
#clusters_means = k_means.cluster_centers_.squeeze()
X_clustered = gmm.predict(X)

X_clustered.shape = img.shape
print(X_clustered.shape)
print(X_clustered[:,:,1])
X_segmented=X_clustered[:,:,0]+X_clustered[:,:,1]+X_clustered[:,:,2]
X_clustered=(255*X_clustered).astype(np.uint8)
io.imsave('aha.png',X_clustered)

plt.imshow(X_segmented, cmap=plt.get_cmap('viridis')) #or another colormap that you like https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()
