import numpy as np
from PIL import Image
import sys
from timeit import default_timer as timer

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os

model_name = "nice_model"
filename = sys.argv[1]

json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

def  calculate_belongship(v):
  x = loaded_model.predict(v,batch_size=1)
  return x

# padding:10
# mas_width: 320
# 

def count_block(img_sample):
  #data = np.asarray(img_sample)
  data = img_sample
  total = 0
  for i in range(0,len(data)):
    total += sum([1 for x in data[i] if x==0])  # search for foreground pixels
  return total

def get_characteristic_vector(img,sample_size):
  v = list()
  width = len(img)
  height = len(img)
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
      #img_sample = img.crop((i_topleft,j_topleft,i_bottomright,j_bottomright))
      img_sample = img[j_topleft:j_bottomright,i_topleft:i_bottomright]
      c = count_block(img_sample)
      v.append(c)
      i_topleft += sample_size
      i_bottomright += sample_size
    j_topleft += sample_size
    j_bottomright += sample_size
  
  return v

def get_roi_value(img_name, sample_size, padding, stepsize):
  img = Image.open(img_name).convert("L")
  max_width = img.size[0]-padding
  max_height = img.size[1]-padding
  # stepsize tiene que ser un numero divisor comun entre max_width y sample_size 
  roi_belongship = np.full((max_height+padding, max_width+padding), 0.)
  raw_values = np.array(img)
  #print([x for x in raw_values])
  #print(raw_values)
  
  i_topleft=padding
  j_topleft=padding
  i_bottomright = padding+sample_size
  j_bottomright = padding+sample_size
  
  # mientras que i_topleft+stepsize + sample_size < max_height
  while(j_bottomright < max_height):
    i_topleft = padding
    i_bottomright = padding + sample_size
    while i_bottomright < max_width:
      block = raw_values[j_topleft:j_bottomright,i_topleft:i_bottomright]
      v = get_characteristic_vector(block,8)
       
      v = np.array(v)
      v=v.reshape((-1,256))
      
      b = calculate_belongship(v) # ann
      
      belongship= np.full((sample_size,sample_size),b)
      roi_belongship[j_topleft:j_bottomright,i_topleft:i_bottomright]+=belongship
      #roi_belongship[j_topleft:j_bottomright,i_topleft:i_bottomright] /= 2
      #print(belongship)
      #print(roi_belongship[j_topleft:j_bottomright,i_topleft:i_bottomright])
      i_topleft += stepsize
      i_bottomright += stepsize
    j_topleft += stepsize
    j_bottomright += stepsize
  print("maximun value: ")
  print(np.amax(roi_belongship))
  print("Maximun value index")
  i,j = np.unravel_index(roi_belongship.argmax(), roi_belongship.shape)
  end = timer()
  print(str(i)+" "+str(j))
  mask = mask_roi(roi_belongship, 20)
  roi_img = filter_image_by_mask(raw_values,mask)
  roi_spectrum = Image.fromarray(roi_belongship).convert("L")
  roi_spectrum.show("roi spectrum")
  roi_img.show("filtered image")
  roi_img.save("roi_images/"+filename[:-4]+"_roi.png")
  roi_spectrum.save("roi_images/"+filename[:-4]+"_filter.png")
  return end
  
def mask_roi(array_img, threshold):  
  #array_img[center_j-b:center_j + b, center_i-b:center_i+b] = np.full((block_size,block_size),255)
  low_values = array_img < threshold
  return low_values

def filter_image_by_mask(raw_img, mask):
  raw_img[mask] = 255
  im = Image.fromarray(raw_img)
  return im

start = timer()
end = get_roi_value(filename,128,15,20)

print("elapsed time: ")
print(end - start)
