import numpy as np
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth

# Read image
im = io.imread('brain_erode_opening.jpg')

# Make the feature vectors
X = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
bandwidth = estimate_bandwidth(X, quantile=0.4, n_samples=500)
# Perform Clustering
N_clus = 3
km = KMeans(3)
km.fit(X.astype(float)) # the .astype method is only to stop the .fit method
# from throwing a warning.
labels = np.reshape(km.labels_, im.shape[0:2])

# Plotting results
pl.figure()
pl.imshow(im)
for l in range(N_clus):
    pl.contour(labels == 2, contours=1, \
               colors=[pl.cm.spectral(l / float(N_clus)), ])
pl.xticks(())
pl.yticks(())
pl.show()
