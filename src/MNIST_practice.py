#%%
import pandas as pd
import numpy as np

from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt




#%%
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(train_X.shape, train_y.shape)
print(test_X.shape, test_y.shape)






# %%
