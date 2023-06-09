import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
import keras
import time
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D,MaxPooling2D
from keras import regularizers
from keras.layers import Flatten,BatchNormalization,Dropout
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


trainset = '/datasets/attack dataset/train-images-idx3-ubyte'
trainlabels = '/datasets/attack dataset/train-labels-idx1-ubyte'

testset = '/datasets/attack dataset/test-images-idx3-ubyte'
testlabels = '/datasets/attack dataset/test-labels-idx1-ubyte'

X_train_attack= idx2numpy.convert_from_file(trainset)
y_train_attack = idx2numpy.convert_from_file(trainlabels)

X_test_attack= idx2numpy.convert_from_file(testset)
y_test_trf = idx2numpy.convert_from_file(testlabels)

nb_classes = 7

X_valid_trf, X_train_trf = X_train_attack[:5000] / 255.0, X_train_attack[5000:] /255.0          #normalization with MinMax
X_test_trf = X_test_attack/ 255.0

y_valid_trf, y_train_trf = y_train_attack[:5000], y_train_attack[5000:]

y_train_trf_categorical = tf.keras.utils.to_categorical(y_train_trf, nb_classes)
y_valid_trf_categorical = tf.keras.utils.to_categorical(y_valid_trf, nb_classes)

with open('./new_network_model.h5', 'rb') as f:
      base_model = pickle.load(f)

learning_rates = [0.1, 0.2, 0.3]

for lr in learning_rates:
  base_model_clone = keras.models.clone_model(base_model)
  base_model_clone.set_weights(base_model.get_weights())
  tr_model = Sequential(base_model_clone.layers[:-1])
  tr_model.add(Dense(100, activation='relu'))
  tr_model.add(Dense(7, activation="softmax"))

  for layer in tr_model.layers[:-1]:
    layer.trainable = False
  optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, decay=0.01)
  tr_model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])  
  start = time.time()
  tr_model.fit(X_train_trf,y_train_trf_categorical , validation_data=(X_valid_trf,y_valid_trf_categorical), batch_size=5, epochs = 50)
  end = time.time()
  time_duration = end-start
  print("Program finishes in {} seconds:".format(time_duration))
  y_test_pred = np.argmax(tr_model.predict(X_test_trf), axis=-1)

  print('Learning rate: ', lr, "test Accuracy: ", accuracy_score(y_test_trf, y_test_pred))
  print(sklearn.metrics.classification_report(y_test_trf, y_test_pred, digits=3))
