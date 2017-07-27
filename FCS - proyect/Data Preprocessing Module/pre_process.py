# Preprocess NIST DATABASE 4 fingerprint classes files
# output: csv file containing paths and classes of 4000 fingerprints in database

infile = open("sd04_md5.lst")

l_paths = list()
for line in infile:
  path = line.split()[1]
  l_paths.append(path)

files = [l_paths[i] for i,x in enumerate(l_paths) if i%2!=0]
images = [l_paths[i] for i,x in enumerate(l_paths) if i%2==0]

print(len(files))
print(len(images))

genders = list()
fingerprint_class = list()
for f in files:
  infile2 = open(f,"r")
  lines = infile2.readlines()
  gender = lines[0].strip().replace(" ","").split(":")[1]
  f_class = lines[1].strip().replace(" ","").split(":")[1]
  genders.append(gender)
  fingerprint_class.append(f_class)
  infile2.close()
  
print(len(genders))
print(len(fingerprint_class))
outfile = open("fingerprint_database.csv","w")

for i in range(0,len(images)):
  outfile.write(images[i]+ ","+fingerprint_class[i] +","+ genders[i]+ "\n")

infile.close()
outfile.close()
