# -*- coding: utf-8 -*-

import numpy as np
import cv2
from skimage import color,io,img_as_ubyte
from matplotlib import pyplot as plt
from skimage.filters import sobel,roberts,prewitt,gaussian
from skimage.filters import threshold_otsu
from scipy import ndimage as nd
import pandas as pd


image=io.imread('fig1.bmp',0)
plt.imshow(image,cmap='gray')
plt.show()
image_gray=img_as_ubyte(color.rgb2gray(image))
plt.imshow(image_gray,cmap='gray')
plt.show()

# plt.hist(image_gray.flat,bins=1000,range=(0,255))
# plt.show()
thresh=threshold_otsu(image_gray)
print(thresh)
sig1=img_as_ubyte((image_gray<=thresh))
sig2=img_as_ubyte((image_gray>thresh))
plt.imshow(sig1,cmap='gray')
plt.show()

kernel=np.ones([3,3])

sobel_img=sobel(sig2)
plt.imshow(sobel_img,cmap='gray')
plt.show()
sobel_img1=sobel_img.reshape(-1)

roberts_img=roberts(sig2)
plt.imshow(roberts_img,cmap='gray')
plt.show()

sum=0;
r,c=sobel_img.shape

for i in range (1,r):
    for j in range (1,c):
        sum=sum+sobel_img[i,j]

normality=sum/(r*c)
print('Normality',normality)

opening=nd.binary_opening(sig1,kernel)
plt.imshow(opening,cmap='gray')
plt.show()

sure_bg=nd.binary_closing(opening, kernel, iterations=2)
plt.imshow(sure_bg,cmap='gray')
plt.show()

sure_fg=nd.binary_opening(opening, kernel, iterations=2)
plt.imshow(sure_fg,cmap='gray')
plt.show()

dist=nd.distance_transform_edt(opening, cv2.DIST_L2, 6)
plt.imshow(dist,cmap='gray')
plt.show()

ret,tt=cv2.threshold(dist,0.13*dist.max(),255,0)
plt.imshow(tt,cmap='gray')
plt.show()

distance=dist.max();
print('Max Dist',distance)

