import numpy as np 
from keras.datasets import mnist
from sklearn.datasets import load_digits
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from keras.layers import * 
from keras.models import Model 
from keras.optimizers import Adam 
import keras.models
from sklearn.metrics import accuracy_score 
from keras.utils import to_categorical 
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import cv2


a = load_digits()
y = a.target 
x = a.data 



(x_train, y_train), (x_test, y_test) = mnist.load_data()


x__train = x_train
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape)
print(y_train.shape)



#x_train, x_test, y_train, y_test = train_test_split(x_, y_category, train_size = 0.8) 
#inputt_shape = x_.shape[1]

loaded_model = tf.keras.models.load_model('my_model3.keras')


input = Input (shape = (784,)) 
dense1 = Dense(512, activation = 'relu') (input) 
dense2 = Dense(256, activation = 'relu') (dense1) 
dense3 = Dense(128, activation = 'relu')(dense2) 
dense4 = Dense(64, activation = 'relu')(dense3) 
dense5 = Dense(32, activation = 'relu')(dense4) 
output = Dense(10, activation = 'softmax') (dense5) 
model = Model(inputs = input, outputs = output) 

model.compile(optimizer = Adam(learning_rate = 0.0005), loss = "categorical_crossentropy", metrics = ['accuracy']) 
history = model.fit(x_train, y_train, epochs = 5, shuffle = True, batch_size = 32, verbose = 1, validation_split = 0.2)

predictions = model.predict(x_test) 
correct = 0 
incorrect = 0 
aa = []

for i in range(len(predictions)): 
    max = np.argmax(predictions[i]) 
    aa.append([max, np.argmax(y_test[i])])
    real = np.argmax(y_test[i])
    if max == real: 
        correct += 1 
    else: 
        incorrect += 1 

print(predictions.shape, y_test.shape)
aa = np.asarray(aa)
print(accuracy_score(aa[:,0], aa[:,1]))
print("correct: ", correct, "incorrect: ",incorrect)
print(history.history)
plt.plot(history.history["accuracy"])
if(accuracy_score(aa[:,0], aa[:,1]) > 0.98):
    model.save('ANO.keras')
else:
    print("aaa")
    open('cisla.py')
''''''



img = cv2.imread('cislo6_2.png')
#cv2.imshow('image', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray scale image', img)
img = cv2.bitwise_not(img)
#cv2.imshow('inverted image', img)
resized = cv2.resize(img, (28,28))
#cv2.imshow('resized image', resized)
#cv2.waitKey(0)
fig,axs = plt.subplots(2)

axs[0].imshow(x__train[0])
axs[1].imshow(resized)


features = resized.reshape(1,-1)

predictions = loaded_model.predict(features)
print(np.argmax(predictions))

plt.show()
