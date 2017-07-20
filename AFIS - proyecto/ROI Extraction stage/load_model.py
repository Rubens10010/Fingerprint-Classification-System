# load json and create model
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
import numpy
import sys
model_name = sys.argv[1]

json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dataset2 = numpy.loadtxt("test.csv", delimiter=",",usecols=range(1,258))
X2 = dataset2[:,0:256]
Y2 = dataset2[:,256]

score = loaded_model.evaluate(X2, Y2, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
