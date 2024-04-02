#Importing librairies

import pandas as pd 
import numpy as np

# Scikit-learn library: For SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm
import itertools

# Matplotlib library to plot the charts
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Library for the statistic data vizualisation
import seaborn

import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
from tensorflow.keras.utils import get_file
import ipaddress
from ipaddress import IPv4Address
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import get_file
import ipaddress
from ipaddress import IPv4Address
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv('./preprocessed_pdfdataset.csv')
#df2 = pd.read_csv('./CTGAN_preprocessed_pdfdataset.csv')

def change_label(data):
  df.Class.replace(['no'],'0',inplace=True)
  df.Class.replace(['yes'],'1',inplace=True) 
change_label(df)
 
df1 =df

df2 = df.loc[range(1,2)]
print (df2)

df2.to_csv("df2.csv",index=False)

# numeric_columns = list(df1.select_dtypes(include=["int32","int64","float32","float64"]).columns)
# Imputer(missing_values=np.nan,strategy="mean")
# minmax = MinMaxScaler()
# for c in numeric_columns:
#     df1[c] = minmax.fit_transform(np.array(df1[c]).reshape(-1,1))




#train, test = train_test_split(df, test_size=0.25, random_state=42)
#train2, test2 = train_test_split(df2, test_size=0.25, random_state=42)
print(df)
df_corr = df.corr() # Calculation of the correlation coefficients in pairs, with the default method:
print (df_corr.Class)
rank = df_corr['Class'] # Retrieving the correlation coefficients per feature in relation to the feature class
df_rank = pd.DataFrame(rank) 
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False) # Ranking the absolute values of the coefficients

# We seperate ours data in two groups : a train dataset and a test dataset

# First we build our train dataset
df_train_all = df
df_train_1 = df_train_all[df_train_all['Class'] == 1] # We seperate the data which are the frauds and the no frauds
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('In this dataset, we have ' + str(len(df_train_1)) +" frauds so we need to take a similar number of non-fraud")

df_sample=df_train_0.sample(300)
df_train = df_train_1.append(df_sample) # We gather the frauds with the no frauds. 
df_train = df_train.sample(frac=1) # Then we mix our dataset

X_train = df_train.drop(['Class'],axis=1) # We drop the features Time (useless), and the Class (label)
y_train = df_train['Class'] # We create our label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


############################## with all the test dataset to see if the model learn correctly ##################
df_test_all = df2
X_test_all = df_test_all.drop([ 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)


X_train_rank = df_train[df_rank.index[1:11]] # We take the first ten ranked features
X_train_rank = np.asarray(X_train_rank)

############################## with all the test dataset to see if the model learn correctly ##################
X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)

class_names=np.array(['0','1']) # Binary label, Class = 1 (fraud) and Class = 0 (no fraud)

# Function to plot the confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

 
#Model Selection

classifier = svm.SVC(kernel='linear') # We set a SVM classifier, the default SVM Classifier (Kernel = Radial Basis Function)

classifier.fit(X_train, y_train) # Then we train our model, with our balanced data train.


#Testing the model

prediction_SVM_all = classifier.predict(X_test_all) #And finally, we predict our data test.

cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)
print('SVM : ')
print(str(accuracy_score(y_test_all, prediction_SVM_all)))
print(classification_report(y_test_all, prediction_SVM_all))
#Models Rank
classifier.fit(X_train_rank, y_train) # Then we train our model, with our balanced data train.
prediction_SVM = classifier.predict(X_test_all_rank) #And finally, we predict our data test.

cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm,class_names)
 
#Re-balanced class weigh :


classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})
classifier_b.fit(X_train, y_train) # Then we train our model, with our balanced data train.

prediction_SVM_b_all = classifier_b.predict(X_test_all) #We predict all the data set.

cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)


 
#Models Rank

classifier_b.fit(X_train_rank, y_train) # Then we train our model, with our balanced data train.
prediction_SVM = classifier_b.predict(X_test_all_rank) #And finally, we predict our data test.


cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm,class_names)


 

#######################################################################################
####################################################################################
#######################################################################################
####################################################################################
#######################################################################################
####################################################################################
# import packages
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt

 

# for PCA
from sklearn.decomposition import PCA

# For oversampling
from imblearn.over_sampling import SMOTE

# for scaling
from sklearn.preprocessing import MaxAbsScaler

# MLP model package
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Isolation Forest and Random Forest packages
from sklearn.ensemble import IsolationForest, RandomForestClassifier

# for scaling
from sklearn.preprocessing import MaxAbsScaler

# MLP model package
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Isolation Forest and Random Forest packages
from sklearn.ensemble import IsolationForest, RandomForestClassifier

import numpy as np
import pandas as pd



 
# remove rows with any missing values
df = df.dropna()
print('Sample size : ' + str(df.shape[0]))
# As seen below the sample size remains the same proving that their are no missing values

# Let's view the context of the dataset
print(df.describe())

# Let's look at what fraction of the dataset is anomalous
fraud = df[df['Class'] == 1]
print('Number of fraudulent samples : ' + str(fraud.shape[0]))
print('Fraction : '  + str(float(fraud.shape[0]/df.shape[0])))
# Only 0.17% of the data is fraudulent, proving the fact that dataset is highly skewed.
# And training any model on the dataset directly will have a very poor result and take
# a very significant amount of resources.
 

# declare a dictionary to hold the models
models_dict = {}

# Let's define the IsolationForest Model
def IsF (X,y,X2, y2, ratio) :
    
   
    # define the model
    model = IsolationForest(n_estimators = 50,random_state = 1)
    
    # train the model
    model.fit(X, y)
    
    # predict on the test set
    predictions = model.predict(X2)

# Convert the predictions according to problem profile
    predictions[predictions ==  1] = 0
    predictions[predictions == -1] = 1
    
    # evaluate the accuracy and classification report
    print('IsF 1: ' + str(accuracy_score(y2, predictions)))
    print(classification_report(y2, predictions))
    
    # assign this to models_dict
    models_dict['IsF'] = model

    # Let's define the MultiLayer Perceptron
def MLP (X,y,X2, y2, param) :
    
   
    # hyper-parameters
    nodes = param['nodes']
    lrate = param['lrate']
    toler = param['toler']
    batch = param['batch_size']

     
    # define the model
    model = MLPClassifier(hidden_layer_sizes = (nodes,), tol = toler, batch_size = batch, learning_rate_init = lrate,
                          verbose = 0, random_state = 1)
    
    # train the model
    model.fit(X, y)
    
    # predict on the test set
    predictions = model.predict(X2)
    
    # evaluate the accuracy and classification report
    print('MLP 1: ' + str(accuracy_score(y2, predictions)))
    print(classification_report(y2, predictions))
    
    # assign this to models_dict
    models_dict['MLP'] = model
    # Let's define the Random Forest Classifier Model
def RFR (X,y,X2, y2, split) :
    
   
    # define the model
    model = RandomForestClassifier(n_estimators = 100, criterion = 'gini', min_samples_split = split, random_state = 1)
    
    # train the model
    model.fit(X, y)

        # predict on the test set
    predictions = model.predict(X2)
    
    # evaluate the accuracy and classification report
    print('RFR 1: ' + str(accuracy_score(y2, predictions)))
    print(classification_report(y2, predictions))
    
    # assign this to models_dict
    models_dict['RFR'] = model


    #Undersampling
    # First let's evaluate the Undersampling method and view its performance on the test set

# # Now to reduce the imbalance in the dataset, group the rows by their labels
# df = df.sort_values(by = ['Class'], ascending = False).reset_index(drop = True)

# # Divide the dataset into training and testing set
# df_1 = df.iloc[: 492, :].reset_index(drop = True)
# df_2 = df.iloc[492 :, :].reset_index(drop = True)

# # Undersample the training dataset
# new_df = df_2.sample(frac = 0.03)

# # Combine the undersampled training set and test set and shuffle them around
# complete_df = pandas.concat([new_df, df_1]).sample(frac = 1).reset_index(drop = True)


# # Assigning important parameters for the models

# # For Isolation Forest Model
ratio = float(df[df['Class'] == 1].shape[0]/df.shape[0])

# # For Multi Layer Perceptron
param = {
    'nodes' : 170,
    'lrate' : 0.00005,
    'toler' : 0.00001,
    'batch_size' : 100
  }

# # For Random Forest Classifier
split = 2
 
X = np.array(df.drop(['Class'], axis = 1))
y = np.array(df['Class'])
    
X2= np.array(df2.drop(['Class'], axis = 1))
y2 = np.array(df2['Class'])
IsF(X,y,X2, y2, ratio)
MLP(X,y,X2, y2,  param)
RFR(X,y,X2, y2,  split)
 