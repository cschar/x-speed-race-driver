'''
Simple and fast image transforms to mimic:
 - brightness
 - contrast
 - erosion 
 - dilation
'''

import cv2
from PIL import ImageGrab
import numpy as np

# Image data
# image = cv2.imread('imgur.png',0) # load as 1-channel 8bit grayscale
bbox = (0, 0, 800, 600)
pil_image = ImageGrab.grab(bbox=bbox).convert('RGB') 
image = np.array(pil_image)
cv2.imshow('image', image)
maxIntensity = 255.0 # depends on dtype of image data
x = np.arange(maxIntensity) 

# Parameters for manipulating image data
phi = 1
theta = 1

# Increase intensity such that
# dark pixels become much brighter, 
# bright pixels become slightly bright
# newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**3
newImage0 = np.array(newImage0,dtype=np.uint8)

cv2.imshow('newImage0',newImage0)


# Close figure window and click on other window 
# Then press any keyboard key to close all windows
closeWindow = -1
while closeWindow<0:
    closeWindow = cv2.waitKey(1) 
cv2.destroyAllWindows()