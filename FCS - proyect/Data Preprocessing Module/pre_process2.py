# fingerprint pre-processing stage
# 1.- fingerprint histogram equalization
# 2.- fingerprint enhancement
# 3.- fingerprint binarization
# 4.- fingerprint thinning

# This program uses threads for processing fingerprint images in NIST DATABASE 4 in the fingerprint pre-processing step of our algorithm

# None of these algorithms belongs to me

import multiprocessing
from multiprocessing import Pool

infile = open("fingerprint_database.csv","r")
infile2 = open("filenames.txt")
filtered_names = infile2.readlines()

files = list()
fingerprint_class = list()
filtered_names = [f.strip() for f in filtered_names]
#Fingerprint_file,class,gender

for line in infile:
  s = line.split(",")
  fingerprint_class.append(s[1])
  o_name = s[0].split("/")
  o_name = o_name[3]
  if o_name not in filtered_names:
    files.append(s[0])
### divide and conquered

import numpy as np
import cv2
#import numpy as np;
import matplotlib.pylab as plt;
import scipy.ndimage
import sys

from image_enhance import image_enhance

def pre_process(img_name):
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

  #ret,img = cv2.adaptiveThreshold  (img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
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
  #cv2.imwrite("thinned.bmp", skel)}
  o_name = img_name.split("/")
  o_name = o_name[3]
  scipy.misc.imsave('fingerprint_thinned/' + o_name, skel);

print("Preprocessing ...")  
if __name__ == '__main__':
  pool = Pool(multiprocessing.cpu_count())
  pool.map(pre_process,files)
  print("finished")
