import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread('brain4.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap='gray')

# =============================================================================
# plt.hist(gray.ravel(),256)
# plt.show()
# =============================================================================

ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
plt.imshow(thresh ,cmap='gray')
