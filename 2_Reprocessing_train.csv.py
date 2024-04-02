import numpy as np
import pandas as pd
import pickle 
from os import path
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import ipaddress
from ipaddress import IPv4Address
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from keras.layers import Dense # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.models import model_from_json 
from keras.layers import LSTM
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.losses import mse, binary_crossentropy, kl_divergence
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.utils import get_file
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools

 

data1 = pd.read_csv('./train2.csv', header=None)


 
data1['target'] =   1
data1['tfrecord'] =  0
data1.to_csv("train3_4.csv",index=False)