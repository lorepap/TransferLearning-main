import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D,MaxPooling2D
from keras import regularizers
from keras.layers import Flatten,BatchNormalization,Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pickle

trainset = 'datasets/pretrained dataset/train-images-idx3-ubyte'
trainlabels = 'datasets/pretrained dataset/train-labels-idx1-ubyte'

testset = 'datasets/pretrained dataset/test-images-idx3-ubyte'
testlabels = 'datasets/pretrained dataset/test-labels-idx1-ubyte'

X_full = idx2numpy.convert_from_file(trainset)
y_full = idx2numpy.convert_from_file(trainlabels)

X_test = idx2numpy.convert_from_file(testset)
y_test = idx2numpy.convert_from_file(testlabels)

nb_classes = 18   

X_val, X_trains = X_full[:10000] / 255.0,  X_full[10000:] /255.0          #normalization with MinMax
y_val, y_trains = y_full[:10000], y_full[10000:]
X_tests = X_test / 255.0


y_trains_categorical = tf.keras.utils.to_categorical(y_trains, nb_classes)
y_val_categorical = tf.keras.utils.to_categorical(y_val, nb_classes)
y_test_categorical = tf.keras.utils.to_categorical(y_test, nb_classes)


model = Sequential() # create Sequential model

model.add(Conv2D(32, (3,3), input_shape=(28,28,1), padding='same', activation = 'relu')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3,3), padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding='same')) 
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(18, activation = 'softmax')) 

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ['accuracy'])

model.fit(X_trains,y_trains_categorical, validation_data=(X_val,y_val_categorical), batch_size=10, epochs = 25)

y_tests_pred = np.argmax(model.predict(X_tests), axis=-1) 

ConfusionMatrixDisplay.from_predictions(y_test, y_tests_pred)
plt.title("Classification Confusion matrix for test set")
plt.show()

# save model
with open('new_network_model.h5','wb') as f:
    pickle.dump(model,f)
