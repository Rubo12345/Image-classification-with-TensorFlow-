import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

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