from ctgan import CTGAN
from ctgan import load_demo
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
 
real_data12 = pd.read_csv('./pdfdataset.csv')

 
real_data1,real_data = train_test_split(real_data12, test_size=0.01, random_state=42)
print(real_data)
#Reset column names for training set
real_data.columns  = ['obj','endobj','stream','endstream','xref','trailer','startxref','Page','Encrypt','ObjStm','JS','Javascript','AA','OpenAction','AcroForm','JBIG2Decode','RichMedia','Launch','EmbeddedFile','XFA','Colors','Class',]

real_data.head()
discrete_columns = ['obj','endobj','stream','endstream','xref','trailer','startxref','Page','Encrypt','ObjStm','JS','Javascript','AA','OpenAction','AcroForm','JBIG2Decode','RichMedia','Launch','EmbeddedFile','XFA','Colors','Class',]


ctgan = CTGAN(epochs=20)
ctgan.fit(real_data, discrete_columns,)

# Create synthetic data
synthetic_data = ctgan.sample(20000)
synthetic_data.to_csv("CTGAN_pdfdataset.csv",index=False)
synthetic_data.head()