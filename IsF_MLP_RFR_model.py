 # import packages
import numpy as np
import pandas
import seaborn
import matplotlib.pyplot as plt

# For GANs
from tensorflow import keras

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



df = pd.read_csv('./creditcard.csv')

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
def IsF (X, y, ratio) :
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    
    # define the model
    model = IsolationForest(n_estimators = 50, contamination = ratio, random_state = 1)
    
    # train the model
    model.fit(X_train, y_train)
    
    # predict on the test set
    predictions = model.predict(X_test)

# Convert the predictions according to problem profile
    predictions[predictions ==  1] = 0
    predictions[predictions == -1] = 1
    
    # evaluate the accuracy and classification report
    print('IsF : ' + str(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions))
    
    # assign this to models_dict
    models_dict['IsF'] = model

    # Let's define the MultiLayer Perceptron
def MLP (X, y, param) :
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    
    # hyper-parameters
    nodes = param['nodes']
    lrate = param['lrate']
    toler = param['toler']
    batch = param['batch_size']

     
    # define the model
    model = MLPClassifier(hidden_layer_sizes = (nodes,), tol = toler, batch_size = batch, learning_rate_init = lrate,
                          verbose = 0, random_state = 1)
    
    # train the model
    model.fit(X_train, y_train)
    
    # predict on the test set
    predictions = model.predict(X_test)
    
    # evaluate the accuracy and classification report
    print('MLP : ' + str(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions))
    
    # assign this to models_dict
    models_dict['MLP'] = model
    # Let's define the Random Forest Classifier Model
def RFR (X, y, split) :
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
    
    # define the model
    model = RandomForestClassifier(n_estimators = 100, criterion = 'gini', min_samples_split = split, random_state = 1)
    
    # train the model
    model.fit(X_train, y_train)

        # predict on the test set
    predictions = model.predict(X_test)
    
    # evaluate the accuracy and classification report
    print('RFR : ' + str(accuracy_score(y_test, predictions)))
    print(classification_report(y_test, predictions))
    
    # assign this to models_dict
    models_dict['RFR'] = model


    #Undersampling
    # First let's evaluate the Undersampling method and view its performance on the test set

# Now to reduce the imbalance in the dataset, group the rows by their labels
df = df.sort_values(by = ['Class'], ascending = False).reset_index(drop = True)

# Divide the dataset into training and testing set
df_1 = df.iloc[: 492, :].reset_index(drop = True)
df_2 = df.iloc[492 :, :].reset_index(drop = True)

# Undersample the training dataset
new_df = df_2.sample(frac = 0.03)

# Combine the undersampled training set and test set and shuffle them around
complete_df = pandas.concat([new_df, df_1]).sample(frac = 1).reset_index(drop = True)


# Assigning important parameters for the models

# For Isolation Forest Model
ratio = float(complete_df[complete_df['Class'] == 1].shape[0]/complete_df.shape[0])

# For Multi Layer Perceptron
param = {
    'nodes' : 170,
    'lrate' : 0.00005,
    'toler' : 0.00001,
    'batch_size' : 100
}

# For Random Forest Classifier
split = 2

# Dividing the dataset into feature matrix and class vector
X = np.array(complete_df.drop(['Class'], axis = 1))
y = np.array(complete_df['Class'])

IsF(X, y, ratio)
MLP(X, y, param)
RFR(X, y, split)




# shuffle the original dataframe and reset the indexes
df = df.sample(frac = 1).reset_index(drop = True)

# create a new dataframe which is a 50% the size of df
test = df.sample(frac = 0.5).reset_index(drop = True)

# create respective numpy arrays
X_original = np.array(test.drop(['Class'], axis = 1))
y_original = np.array(test['Class'])

# calculate the performance of models on these sets
# IsF
pred_1 = models_dict['IsF'].predict(X_original)
pred_1[pred_1 ==  1] = 0
pred_1[pred_1 == -1] = 1
print('IsF : ' + str(accuracy_score(y_original, pred_1)))
print(classification_report(y_original,pred_1))

# MLP
pred_2 = models_dict['MLP'].predict(X_original)
print('MLP : ' + str(accuracy_score(y_original, pred_2)))
print(classification_report(y_original,pred_2))

# RFR
pred_3 = models_dict['RFR'].predict(X_original)
print('RFR : ' + str(accuracy_score(y_original, pred_3)))
print(classification_report(y_original,pred_3))

#Oversampling 

# Extract from the dataframe, class 1s and 0s
df_1 = df[df['Class'] == 1].sample(frac = 1.0).reset_index(drop = True)
df_2 = df[df['Class'] == 0].sample(frac = 1.0).reset_index(drop = True)

# Split each dataframe to certain fraction
new_df_1, old_df_1 = df_1[: 480].reset_index(drop = True), df_1[480 :].reset_index(drop = True)
new_df_2, old_df_2 = df_2[: 999].reset_index(drop = True), df_2[999 :].reset_index(drop = True)



# group them into test and train sets
test, train = pandas.concat([new_df_1, new_df_2]), pandas.concat([old_df_1, old_df_2])
test, train = test.sample(frac = 1.0).reset_index(drop = True), train.sample(frac = 1.0).reset_index(drop = True)

# Now divide train dataframe into 'X' and 'y'
X = np.array(train.drop(['Class'], axis = 1))
y = np.array(train['Class'])

# Oversample on this set
oversample = SMOTE(sampling_strategy = 0.4, random_state = 1, k_neighbors = 5)

# Get new feature matrix and class vector
X_over, y_over = oversample.fit_resample(X, y)


# Assigning important parameters for the models

# For Isolation Forest Model
ratio = float(y_over[y_over == 1].shape[0]/y_over.shape[0])

# For Multi Layer Perceptron
param = {
    'nodes' : 170,
    'lrate' : 0.00005,
    'toler' : 0.00001,
    'batch_size' : 100
}

# For Random Forest Classifier
split = 2

IsF(X_over, y_over, ratio)
MLP(X_over, y_over, param)
RFR(X_over, y_over, split)

# Divide test dataframe into 'X' and 'y'
X_original = np.array(test.drop(['Class'], axis = 1))
y_original = np.array(test['Class'])

# Test it on all the three models
# IsF
pred_1 = models_dict['IsF'].predict(X_original)
pred_1[pred_1 ==  1] = 0
pred_1[pred_1 == -1] = 1
print('IsF : ' + str(accuracy_score(y_original, pred_1)))
print(classification_report(y_original,pred_1))


# MLP
pred_2 = models_dict['MLP'].predict(X_original)
print('MLP : ' + str(accuracy_score(y_original, pred_2)))
print(classification_report(y_original,pred_2))

# RFR
pred_3 = models_dict['RFR'].predict(X_original)
print('RFR : ' + str(accuracy_score(y_original, pred_3)))
print(classification_report(y_original,pred_3))



