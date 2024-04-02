import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import tensorflow as tf

import os
import re
import seaborn as sns
import numpy as np
import pandas as pd
import math
from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as L
#import tensorflow_addons as tfa
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

#import efficientnet.tfkeras as efn

#from kaggle_datasets import KaggleDatasets
from tensorflow.keras import backend as K
from numpy.random import randn
from tensorflow.keras.models import load_model 
from matplotlib import pyplot
import matplotlib.pyplot as plt


# img_gen_model = load_model('./model_path_VGG19/model_path3/vgg19_generator_model_192.h5',compile=False, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()})
# vector =np.asarray([[0.5 for _ in range(4096)]])
#     # specify the label
#     # predict on generator
# X= img_gen_model.predict(vector)
#      # #scale from [-1,1] to [0,1]
# X = (X + 1) / 2
#     # return fraud samples             
# plt.imshow(X[0, :, :])
# plt.show()

 

 
# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot

image_path = "./VGG19_GAN_image_192/1/"
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
 # generate points in the latent space
 x_input = randn(latent_dim * n_samples)
 # reshape into a batch of inputs for the network
 x_input = x_input.reshape(n_samples, latent_dim)
 return x_input
 
# plot the generated images
def create_plot(examples, n):
 # plot images
 for i in range(n):
 # define subplot
    #pyplot.subplot(n, n, 1 + i)
 # turn off axis
    #pyplot.axis('off')
 # plot raw pixel data
    plt.imshow(examples[i, :, :])
    plt.savefig(image_path+str(i)+'.png')  
    #plt.show()
 
# load model
model = load_model('./model_path_VGG19/model_path1/vgg19_generator_model_192.h5',compile=False, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU()})
#model = load_model('./model_path_VGG19/model_path0/vgg19_generator_model_192.h5')
# generate images
latent_points = generate_latent_points(4096, 250)
# generate images
X = model.predict(latent_points)
# scale from [-1,1] to [0,1]
X = (X + 1) / 2.0
# plot the result
create_plot(X, 250)


