# CNN Filters application

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.ndimage.filters import convolve
from skimage import color,io

# Importing color image as greyscale
img = color.rgb2gray(io.imread('Barry_Chuckle_Everyday.jpg'))


# Edge detection filter
Edge_detection_filter = np.array([
                            [0,1,0],
                            [1,-4,1],
                            [0,1,0]])
img_edg_detect = convolve(img,Edge_detection_filter)


# Sharpen filter
Sharpen_filter = np.array([
                        [0,-1,0],
                        [-1,5,-1],
                        [0,-1,0]])
img_sharpen = convolve(img,Sharpen_filter)


# Gaussian blur filter
Gaussian_blur_filter = (1/16) *np.array([
                        [1,2,1],
                        [2,4,2],
                        [1,2,1]])
img_gauss_blur = convolve(img,Gaussian_blur_filter)


# Plotting the images
plt.figure(figsize=(8,8))

# Original Image
plt.subplot(2,2,1)
plt.imshow(img,cmap=cm.gray)
plt.title('Original image')

# Edge detection
plt.subplot(2,2,2)
plt.imshow(img_edg_detect,cmap=cm.gray)
plt.title("Edge detection")

# Sharp Image
plt.subplot(2,2,3)
plt.imshow(img_sharpen,cmap=cm.gray)
plt.title("Sharp Image")

# Gaussian Blur
plt.subplot(2,2,4)
plt.imshow(img_gauss_blur,cmap=cm.gray)
plt.title("Gaussian Blur")

plt.show()


print("Completed!")


