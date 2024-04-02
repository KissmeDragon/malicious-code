import pandas as pd
import numpy as np
from tensorflow.keras.utils import get_file
import ipaddress
from ipaddress import IPv4Address
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer as Imputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


 
import numpy as np
import pandas as pd



df = pd.read_csv('./CTGAN_pdfdataset.csv')
 

 




def change_label(data):
  df.Class.replace(['no'],'0',inplace=True)
  df.Class.replace(['yes'],'1',inplace=True) 
change_label(df)

df1=df
df1.to_csv("CTGAN_preprocessed_pdfdataset2.csv",index=False)

#Normalization dataset by using MinMaxScaler technique
numeric_columns = list(df.select_dtypes(include=["int32","int64","float32","float64"]).columns)
Imputer(missing_values=np.nan,strategy="mean")
minmax = MinMaxScaler()
for c in numeric_columns:
    df[c] = minmax.fit_transform(np.array(df[c]).reshape(-1,1))


#Split dataset into 2 parts Train and Test, after that writing them to 2 file .csv
#train, test = train_test_split(df, test_size=0.5, random_state=42)


#print('Train shape: {}'.format(train.shape))
#print('Test shape: {}'.format(test.shape))
#210600.
df.to_csv("CTGAN_preprocessed_pdfdataset.csv",index=False)
# train.to_csv("PDFMalware2022_train.csv",index=False)
# test.to_csv("PDFMalware2022_test.csv",index=False)
 
 