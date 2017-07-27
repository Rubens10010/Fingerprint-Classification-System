#!/usr/bin/env python3

# test with block size: 10 y 20
# input images: 200x200
# output orientation field vector = 100 y 400

import multiprocessing
from multiprocessing import Pool
from functools import partial

import numpy as np
import scipy.ndimage as ndimage
import sys
import utils
from timeit import default_timer as timer

db= sys.argv[1]

fn = "om_featuresDB2.dat"
create_fn = open(fn,"w+")
create_fn.close()

infile = open(db, "r")
files = list()

for line in infile:
  line = line.strip()
  files.append(line)  # path
infile.close()

def get_features_vector(q,sourceImage):
    #print(sourceImage)
    sourceImage = sourceImage.split(" ")
    filename = sourceImage[0]
    class_img = sourceImage[1]
    np.set_printoptions(
            threshold=np.inf,
            precision=4,
            suppress=True)

    image = ndimage.imread(filename, mode="L").astype("float64")
    orientations = utils.estimateOrientations(image,w=20, interpolate=False)
    
    with open(fn, 'r') as f:
      size = len(f.read())
    #print(str(orientations.flatten()))
    res = filename + " "
    for feature in orientations:
      res += str(feature) + " "
    res += str(class_img)
    q.put(res)
    #return orientations

#print(get_features_vector(sourceImage))

def listener(q):
  ''' listens for messages on the q, writes to file '''
  f = open(fn, 'wb')
  while 1:
    m = q.get()
    if m == "kill":
      print("Feature vector file closed");
      break
    f.write(str(m)+'\n')
    f.flush()
  f.close()

print("Calculating Roi's....")
if __name__ == '__main__':
  p_inicio = timer()
  manager = multiprocessing.Manager()
  q = manager.Queue()
  pool = Pool(multiprocessing.cpu_count())
  
  func = partial(get_features_vector, q)
  watcher = pool.apply_async(listener, (q,))
  pool.map(func, files)
  
  q.put("kill")
  pool.close()
  pool.join()
  p_fin = timer()
  print("finished in: ")
  print(p_fin - p_inicio)
  
