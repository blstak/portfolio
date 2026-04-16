import numpy as np 
from sklearn.datasets import load_wine 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from keras.layers import * 
from keras.models import Model 
from keras.optimizers import Adam 
from sklearn.metrics import accuracy_score 
from keras.utils import to_categorical 
import matplotlib.pyplot as plt

a = load_wine() 
y = a.target 
x = a.data 

sc = MinMaxScaler(feature_range=(-1 ,1)) 
x_ = sc.fit_transform(x) 
y_category = to_categorical (y) 

x_train, x_test, y_train, y_test = train_test_split(x_, y_category, train_size = 0.8) 

input = Input (shape = (x_.shape[1])) 
dense1 = Dense(100, activation = 'selu') (input) 
dense2 = Dense(80, activation = 'tanh')(dense1) 
dense3 = Dense(40, activation = 'selu')(dense2) 
dense4 = Dense(15, activation = 'tanh')(dense3) 
output = Dense(3, activation = 'softmax') (dense4) 
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
print(x)
print(y)
print(accuracy_score(aa[:,0], aa[:,1]))
print("correct: ", correct, "incorrect: ",incorrect)
print(history.history)
plt.plot(history.history["accuracy"])
plt.show()
