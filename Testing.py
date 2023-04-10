import numpy as np
import pandas as pd
import keras
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import numpy as np

df=pd.read_table('Test.txt',sep='\t')
rows=df.index.values.tolist()
rows=np.array(rows)
X_list=df.iloc[:,0:-1]
X=np.array(X_list)
w=open('Predict.txt','w')
model=load_model(r'DNN.model')
y_pred1 = model.predict(X)
for i,k in enumerate(rows):
    w.write(rows[i]+'\t'+'\t'.join([str(x) for x in y_pred1[i][:]])+'\n')
