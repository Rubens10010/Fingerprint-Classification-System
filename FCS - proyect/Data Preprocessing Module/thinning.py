import cv2
import numpy as np

if(len(sys.argv)<2):
    img_name = 'test.png'
elif(len(sys.argv) >= 2):
    img_name = sys.argv[1];
img = cv2.imread(img_name,0) # works in images without noise
print(img)
size = np.size(img)
skel = np.zeros(img.shape,np.uint8)
 
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
cv2.imshow("skel",skel)
#print(skel[100])
cv2.imwrite("thinned_" +img_name, skel)
cv2.waitKey(0)
cv2.destroyAllWindows()
