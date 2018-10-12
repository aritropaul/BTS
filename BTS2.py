import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
import scipy as sp
import pylab as pl
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import KMeans
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth


def ShowImage(title,img,ctype):
    plt.figure(figsize=(10, 10))
    if ctype=='bgr':
        b,g,r = cv2.split(img)       # get b,g,r
        rgb_img = cv2.merge([r,g,b])     # switch it to rgb
        plt.imshow(rgb_img)
    elif ctype=='hsv':
        rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        plt.imshow(rgb)
    elif ctype=='gray':
        plt.imshow(img,cmap='gray')
    elif ctype=='rgb':
        plt.imshow(img)
    else:
        raise Exception("Unknown colour type")
    plt.axis('off')
    plt.title(title)
    plt.show()

img = cv2.imread('/Users/aritropaul/Desktop/notumorbrain.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ShowImage('Brain with Skull',gray,'gray')

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
ShowImage('Applying Otsu',thresh,'gray')
colormask = np.zeros(img.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255))
blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
ShowImage('Blended', blended, 'bgr')
ret, markers = cv2.connectedComponents(thresh)

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
#Get pixels which correspond to the brain
brain_mask = markers==largest_component

brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
ShowImage('Connected Components',brain_out,'rgb')

brain_mask = np.uint8(brain_mask)
kernel = np.ones((8,8),np.uint8)
closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
ShowImage('Closing', closing, 'gray')
brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[closing==False] = (0,0,0)
ShowImage('Connected Components',brain_out,'rgb')
cv2.imwrite('brain_final.png',brain_out)

img1 = cv2.imread('brain_final.png',0)
ret,thresh1 = cv2.threshold(img1,155,255,cv2.THRESH_BINARY)
plt.imshow(thresh1)
(_,cnts,_) = cv2.findContours(thresh1.copy(), cv2.RETR_TREE,
                              cv2.CHAIN_APPROX_SIMPLE)
print(str(len(cnts))+' contours detected')

# find maximum area contour
if (len(cnts) > 0):
    area = np.array([cv2.contourArea(cnts[i]) for i in range(len(cnts))])
    #list of all areas
    maxa_ind = np.argmax(area)
    xx = [cnts[maxa_ind][i][0][0] for i in range(len(cnts[maxa_ind]))]
    yy = [cnts[maxa_ind][i][0][1] for i in range(len(cnts[maxa_ind]))]
    #ROI.append([min(xx),max(xx),min(yy),max(yy)])
    plt.imshow(img1)
    plt.plot(xx,yy,'r',linewidth=3)
    plt.title('Tumor Area')
    #plt.hist(img.ravel(),256,[0,256]);
    plt.show()
else:
    print("No Tumors found!")
