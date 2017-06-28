import numpy as np
import cv2

from matplotlib import pyplot as plt

#cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel 
#canhold

g_kernel = cv2.getGaborKernel((25, 25), 6.0, np.pi/4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
g_kernel1 = cv2.getGaborKernel((30, 30), 6.0, (3*np.pi)/4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
g_kernel2 = cv2.getGaborKernel((30, 30),4 , 0, 8, 0.5, 0, ktype=cv2.CV_32F)
g_kernel3 = cv2.getGaborKernel((30, 30),4 , np.pi, 8, 0.5, 0, ktype=cv2.CV_32F)

print np.pi/4
img = cv2.imread('ppf1_1.png')
img1 = cv2.imread('ppf1_1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Otsu thresholding
ret2,img1 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('otsu', img1)

filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
filtered_img1 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel1)
filtered_img2 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel2)
filtered_img3 = cv2.filter2D(img, cv2.CV_8UC3, g_kernel3)

cv2.imshow('0', filtered_img)
cv2.imshow('1', filtered_img1)
cv2.imshow('2', filtered_img2)
cv2.imshow('image', img)

cv2.addWeighted(filtered_img2,0.4,filtered_img1,0.8,0,img) #0 degree and 90
cv2.addWeighted(img,0.4,filtered_img,0.6,0,img) #0 degree and 90
cv2.addWeighted(img,0.4,filtered_img3,0.6,0,img)
cv2.addWeighted(img,0.4,img1,0.6,0.3,img)

cv2.imshow('per',img)

#threshold will convert it plain zero and white image
ret,thresh1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)#127 instead of 200

cv2.imshow('per1',thresh1)

h, w = g_kernel.shape[:2]
g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
g_kernel1 = cv2.resize(g_kernel1, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)

cv2.imshow('gabor kernel (resized)', g_kernel)
cv2.imshow('gabor kernel1 (resized)', g_kernel1)

cv2.waitKey(0)
cv2.destroyAllWindows()
