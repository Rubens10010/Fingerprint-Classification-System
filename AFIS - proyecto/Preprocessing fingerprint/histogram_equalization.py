import numpy as np
import cv2
import sys

if(len(sys.argv)<2):
    img_name = 'test.png'
elif(len(sys.argv) >= 2):
    img_name = sys.argv[1];
img = cv2.imread(img_name,0)
print(img)
  
 # create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
  
cv2.imwrite('equalized_' + img_name,cl1)
