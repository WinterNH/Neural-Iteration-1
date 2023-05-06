import pandas as pd
import numpy as np
from tensorflow import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt


(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

data = pd.read_csv


for i in range(9):  
    plt.subplot(330 + 2 + i)
    plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.show()