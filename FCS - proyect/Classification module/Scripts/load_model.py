#test model
import numpy as np
import sys
from timeit import default_timer as timer

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from keras.utils import np_utils
from timeit import default_timer as timer

model_name = sys.argv[1]
input_number = 400
nb_classes = 5

# loading features dataset
dataset = numpy.loadtxt("om_featuresDB2_2.dat", delimiter=" ",usecols=range(1,402))
print("Dataset loaded")

#numpy.random.shuffle(dataset)
#split into input (x) and output y variables
#print(sdf)
X = dataset[:,0:input_number].astype('float32')
Y = dataset[:,input_number].astype('float32')

#X = (X-X.min(0))/X.ptp(0)

x_train = X[:2000,]
x_test = X[2000:,]
y_train = Y[:2000,]
y_test = Y[2000:,]

# normalize train and test
minimun = x_train.min(0)
diff = x_train.ptp(0)
x_train = (x_train - minimun)/diff
x_test = (x_test - minimun)/diff

print('Train samples: {}'.format(x_train.shape[0]))
print('Test samples: {}'.format(x_test.shape[0]))

#x_test = x_train
#y_test = y_train

# categorize train and test output
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

start = timer()

json_file = open(model_name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_name+".h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data4
#loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
end = timer()
print("loaded and compiled: ")
elapsed = end - start

score = loaded_model.evaluate(x_test, y_test, verbose=1)
print("\n"+str(score[0]) + " accuracy: " + str(score[1]) + " " + str(elapsed))
