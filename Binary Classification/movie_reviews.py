import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#preprocessing the data as vectors
def vectorize(seq, dim=10000):
	res = np.zeros((len(seq), dim))
	for i, seq in enumerate(seq):
		res[i, seq] = 1
	return res

x_train = vectorize(train_data)
x_test = vectorize(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#defining the model of the network
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)

print(results)


