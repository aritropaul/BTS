import cv2
import numpy as np
from matplotlib import pyplot as plt

img_ini = cv2.imread('brain.jpeg')
gray_img = cv2.cvtColor(img_ini,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img,cmap='gray')
plt.title("initial image")
plt.show()

#apply median filter to smoothen image
gray = cv2.medianBlur(gray_img,3) 
cv2.imwrite("Median_filter.jpg",gray)
plt.imshow(img_ini,cmap="gray")
plt.title("Image after applying medium filter")
plt.show()

#applying thresholding  to get the skull portion
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU) 
cv2.imwrite("Threshold_img.jpg",thresh)
plt.imshow(thresh ,cmap='gray')
plt.show()

#apply mask
colormask = np.zeros(img_ini.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255)) #overlaying mask over original image
blended = cv2.addWeighted(img_ini,0.7,colormask,0.1,0)
b,g,r = cv2.split(blended)
rgb_img = cv2.merge([r,g,b])
plt.imshow(rgb_img)
plt.show()
cv2.imwrite("blended_img.jpg",rgb_img)

#finding the connected components in the image
ret, markers = cv2.connectedComponents(thresh)
plt.imshow(markers)
plt.show()
cv2.imwrite("Connected_component_img.jpg",markers)

#finding the largest one which will be the brain  
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
largest_component = np.argmax(marker_area)+1                    
brain_mask = markers==largest_component
brain_out = img_ini.copy()
brain_out[brain_mask==False] = (0,0,0) #filling rest of the background with black
plt.imshow(brain_out)
plt.show()
cv2.imwrite("brain.jpg",brain_out)