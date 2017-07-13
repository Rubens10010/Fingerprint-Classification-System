# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import cv2
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys

from image_enhance import image_enhance

print('loading sample image');
if(len(sys.argv)<2):
    img_name = 'test.tif'
elif(len(sys.argv) >= 2):
    img_name = sys.argv[1];
img = scipy.ndimage.imread(img_name);    ## original image

if(len(img.shape)>2):
    # img = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    img = np.dot(img[...,:3], [0.299, 0.587, 0.114]);
    
# histogram equalization
#img = cv2.imread(img_name,0)

#print(img)
  
 # create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)  ## histogram equalization
  
#cv2.imwrite('clahe.jpeg',cl1)

### Enhancing image ###

rows,cols = np.shape(img);
aspect_ratio = np.double(rows)/np.double(cols);

new_rows = 350;             # randomly selected number
new_cols = new_rows/aspect_ratio;

#img = cv2.resize(img,(new_rows,new_cols));
img = scipy.misc.imresize(img,(np.int(new_rows),np.int(new_cols)));

enhanced_img = image_enhance(img);    
img = enhanced_img*1
img = np.array(img*255,dtype = np.uint8)

# thinning -> skeletonization
#print(img)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)

#ret,img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#cv2.THRESH_BINARY,11,2)
ret,img = cv2.threshold(img,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False
 
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
 
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

skel = cv2.bitwise_not(skel)
#cv2.imwrite("thinned.bmp", skel)

if(1):
    print('saving the thinned image')
    scipy.misc.imsave('Pre-processed/' + img_name, skel);
else:
    plt.imshow(enhanced_img,cmap = 'Greys_r');
