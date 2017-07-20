from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
#from random import shuffle
import os
import numpy

#numpy.random.seed(7)
dataset = numpy.loadtxt("roiDataset.csv", delimiter=",",usecols=range(1,258))
print("training...")

numpy.random.shuffle(dataset)
#split into input (x) and output y variables
#print(sdf)
X = dataset[:,0:256]
Y = dataset[:,256]
print(Y)

#X = X[50:150]
#Y = Y[50:150]
#print(Y)

model = Sequential()
model.add(Dense(32, input_dim=256, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(X)
#print(Y)

# Fit the model
model.fit(X, Y, epochs=100, batch_size=100,verbose=2)

dataset2 = numpy.loadtxt("test.csv", delimiter=",",usecols=range(1,258))
X2 = dataset2[:,0:256]
Y2 = dataset2[:,256]
#print("testing...")
#scores = model.evaluate(X2, Y2)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

print("predicting")
predictions = model.predict(X2)
predictions = [round(x) for x in predictions]
print(predictions)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...
"""
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))"""
