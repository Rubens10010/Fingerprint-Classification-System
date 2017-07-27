import numpy as np
from PIL import Image
from scipy import ndimage
import sys
from timeit import default_timer as timer

database = sys.argv[1] # fingerprint roi dataset.csv

infile = open(database)
lines = list()
for l in infile:
  if "_filter" in l:
    continue
  l = l.strip()
  l = l[2:]
  lines.append(l)

output = open("roi_images_db2.csv","w+")
counter = 0

d = {"A":0,"L":1,"R":2,"T":3 ,"W":4}
start = timer()
print("processing roi region... ")
for l in lines:
  parts = l.split("/")
  img_class = parts[1]
  img_name = parts[2][:8]
  thinned_filename = "classes/"+img_class+"/"+img_name+".png"
  
  roi_map_name = parts[0] +"/"+ img_class+"/" + img_name + "_filter.png"
  
  roi_filter_img = Image.open(roi_map_name).convert("L")
  roi_filter_data = np.asarray(roi_filter_img)
  
  thinned_img = Image.open(thinned_filename).convert("L")
  #maximun = np.amax(roi_filter_data)
  i,j = np.unravel_index(roi_filter_data.argmax(), roi_filter_data.shape)
  
  # get a square center in i.j in thinned_img of 200 x 200
  width, height = thinned_img.size
  new_width = 200 # PARAMETER
  new_height = 200 # PARAMETER
  
  space_left_x = width - i
  space_left_y = height - j
  
  left = i - new_width/2
  top = j - new_height/2
  right = i + new_width/2
  bottom = j + new_height/2
  
  space_x = 0
  space_y = 0
  if new_width/2 > i:
    space_x = new_width - i
    left = 0
    right += space_x
  if new_height/2 > j:
    space_y = new_height - j
    top = 0
    bottom += space_y
  if i + space_left_x > width:
    space_x = (i + space_left_x) - width
    rigth = width
    left -= space_x
  if j + space_left_y > height:
    space_y = (i + space_left_y) - height
    bottom = height
    top -= space_y
  thinned_img = thinned_img.crop((left,top,right,bottom))
  #cropped = Image.fromarray(crop_img.astype('uint8'))
  path = "roi_images_db2/" +img_class+"/" +img_name+"_roi.png"
  thinned_img.save(path)
  counter +=1
  if(counter%2 == 1000): print("Done 1000!")
  output.write(path + " " + str(d[img_class])+ "\n")

end = timer()
print("elapsed time: ")
print(end - start)

# elapsed time for processing with model1 : 153.46307
