#!/usr/bin/env python
import numpy as np
from skimage import io, color
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure
from skimage.transform import resize
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.morphology import extrema


# load dtm image
image = color.rgb2gray(io.imread('2092.jpg'))
xsize=image.shape[0]
ysize=image.shape[1]
#img=resize(img, (int(img.shape[0] / 6), int(img.shape[1] / 6)))

# k-means clustering of the image (population of pixels intensities)
#image = image.reshape((-1, 1))

distance = ndi.distance_transform_edt(image)
local_maxi = extrema.local_maxima(distance, image, np.ones((3, 3)))
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

##distance = ndi.distance_transform_edt(image)
##local_maxi = peak_local_max(
##    distance, indices=False, footprint=np.ones((3, 3)), labels=image)
##markers = measure.label(local_maxi)
##labels_ws = watershed(-distance, markers, mask=image)

print(labels.shape)

X_clustered=labels

X_clustered.shape = image.shape

print(X_clustered.shape)

X_clustered=resize(image, (xsize, ysize))
#print(X_clustered.shape)
#print(X_clustered[:,:,1])
#X_segmented=X_clustered[:,:,0]+X_clustered[:,:,1]+X_clustered[:,:,2]
X_segmented=X_clustered
#X_clustered=resize(img, (xsize, ysize))
X_clustered=(255*X_clustered).astype(np.uint8)
io.imsave('aha.png',X_clustered)

plt.imshow(X_segmented, cmap=plt.get_cmap('viridis')) #or another colormap that you like https://matplotlib.org/examples/color/colormaps_reference.html
plt.show()
