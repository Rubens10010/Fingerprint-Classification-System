import numpy as np
from PIL import Image
import sys
img_name = sys.argv[1];
output = sys.argv[2]
outfile = open("test.csv","a")

img = Image.open(img_name).convert("L")

def count_block(img_sample):
  data = np.asarray(img_sample)
  total = 0
  for i in range(0,len(data)):
    total += sum([1 for x in data[i] if x==0])  # search for foreground pixels
  return total

def get_characteristic_vector(img,sample_size):
  v = list()
  width = img.size[0]
  height = img.size[1]
  i_topleft=0
  j_topleft=0
  i_bottomright = sample_size
  j_bottomright = sample_size
  
  n_blocks = width/sample_size

  while j_bottomright <= height:
    i_topleft=0
    i_bottomright = sample_size
    #j_topleft=0
    while i_topleft != width:
      img_sample = img.crop     ((i_topleft,j_topleft,i_bottomright,j_bottomright))
      c = count_block(img_sample)
      v.append(c)
      i_topleft += sample_size
      i_bottomright += sample_size
    j_topleft += sample_size
    j_bottomright += sample_size
  
  return v

v = get_characteristic_vector(img,8)
outfile.write(img_name+",")
for i in v:
  outfile.write(str(i)+",")
outfile.write(output+"\n")
outfile.close()
