# Use this if fingerprint pre-processing task is stopped during execution

import multiprocessing
from multiprocessing import Pool

infile = open("fingerprint_database.csv","r")
infile2 = open("filenames.txt")
filtered_names = infile2.readlines()
outfile = open("fingerprint_database2.csv","w")

files = list()
fingerprint_class = list()

#Fingerprint_file,class,gender
filtered_names = [f.strip() for f in filtered_names]

for line in infile:
  s = line.split(",")
  fingerprint_class.append(s[1])
  o_name = s[0].split("/")
  o_name = o_name[3]
  files.append(o_name)
  outfile.write("fingerprint_thinned/"+o_name + "," + s[1]+"\n")

#print(filtered_names)
files = [f for f in files if f not in filtered_names]

infile.close()
infile2.close()
outfile.close()
