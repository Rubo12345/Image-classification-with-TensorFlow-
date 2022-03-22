import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# print('Tensorflow Version:', tf.__version__)

(x_train,y_train),(x_test,y_test) = mnist.load_data()  # split the data into train and test

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# plt.imshow(x_train[0], cmap = 'binary')
# plt.show()
# print(y_train[0])
# print(set(y_train))

y_train_one_hot = to_categorical(y_train)  #One hot encoding
y_test_one_hot = to_categorical(y_test)    #One hot encoding

# print(y_train_one_hot.shape)
# print(y_test_one_hot.shape)

# print(y_train_one_hot[0])
# print(y_test_one_hot[0])

x_train_reshape = np.reshape(x_train,(60000,784))  # Converting N -dimensional arrays into vectors
x_test_reshape = np.reshape(x_test,(10000,784))    # Converting N -dimensional arrays into vectors

# print(x_train_reshape.shape)
# print(x_test_reshape.shape)

# print(set(x_train_reshape[0]))   # pixel values

'''
Data Normalization for better computtion and flow.
Finding mean and standard deviation.
Initializing epsilon
x_train_norm = (x_train - x_mean) / (x_std + epsilon)
why epsilon?
To avoid very low values of x_std which might cause disturbance in the data
It is a standard practice during preprocessing
'''

x_mean = np.mean(x_train_reshape)
x_std = np.std(x_train_reshape)
epsilon = 1e-10
x_train_norm = (x_train_reshape - x_mean)/(x_std + epsilon)
x_test_norm = (x_test_reshape - x_mean)/(x_std + epsilon)

# print(x_train_norm[0])
# print(x_test_norm[0])


# Creating a model below
model = Sequential([
    Dense(128,activation='relu',input_shape = (784,)),
    Dense(128,activation='relu'),
    Dense(10,activation = 'softmax')
])

# Compiling the model below

model.compile(
    optimizer='sgd',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
# model.summary()

#Traing the Model
model.fit(x_train_norm,y_train_one_hot,epochs = 3)

#Evaluating the model
loss, accuracy = model.evaluate(x_test_norm,y_test_one_hot)
print('test_set_accuracy',accuracy * 100)

#Predictions
predictions = model.predict(x_test_norm)
print('Shape of Predictions:', predictions.shape)

plt.figure(figsize = (12,12))
start_index = 0
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    prediction = np.argmax(predictions[start_index + i])
    ground_truth = y_test[start_index + i]

    col = 'g'
    if prediction != ground_truth:
        col = 'r'

    plt.xlabel('i= {},pred = {},ground_truth = {}'.format(start_index + i,prediction,ground_truth), color = col)
    plt.imshow(x_test[start_index + i], cmap = 'binary')

plt.show()

plt.plot(predictions[4])
plt.show()