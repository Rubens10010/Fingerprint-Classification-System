import cv2

# clasiffy images from fingerprint_database.csv into files, 1 for each class

infile = open("fingerprint_database.csv")

for line in infile:
  s = line.split(',')
  name = s[0]
  tipe = s[1].strip()
  img = cv2.imread(name)
  cv2.imwrite("classes/"+tipe+"/" + name.split("/")[1],img)
