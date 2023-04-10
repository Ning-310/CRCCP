import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras import Sequential
from keras import optimizers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Softmax,Activation,Dense
from keras.layers import Input, Dropout, Lambda, Dot, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import Callback,ModelCheckpoint
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def DNN_Simple(shape):
    X_in = Input(shape=(shape,))
    H2 = Dense(1024, activation='tanh', kernel_regularizer=l2(5e-3))(X_in)
    H3 = Dropout(0.5)(H2)
    H4 = Dense(512, activation='tanh', kernel_regularizer=l2(5e-3))(H3)
    H5 = Dropout(0.5)(H4)
    H6 = Dense(512, activation='tanh', kernel_regularizer=l2(5e-3))(H5)
    H7 = Dropout(0.5)(H6)
    H8 = Dense(256, activation='tanh', kernel_regularizer=l2(5e-3))(H7)
    H9 = Dropout(0.5)(H8)
    H10 = Dense(128, activation='tanh', kernel_regularizer=l2(5e-3))(H9)
    H11 = Dropout(0.5)(H10)
    Y = Dense(6, activation='softmax')(H11)
    model = Model(inputs=X_in, outputs=Y)
    model.compile(optimizer=optimizers.Adam(lr=0.01, decay=0.98), loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    return model

def Train_fold(i,X_train,y_train,X_val,y_val):
    model=DNN_Simple(9634)
    class RocAucEvaluation(Callback):
        def __init__(self, validation_data=()):
            super(Callback, self).__init__()
            self.x_val,self.y_val = validation_data
        def on_epoch_end(self, epoch, log={}):
            y_pred = self.model.predict(self.x_val)

    RocAuc = RocAucEvaluation(validation_data=(X_val, y_val))
    checkpoint = ModelCheckpoint('./DNN'+str(i+1)+'.model',save_weights_only = False, monitor='val_loss', verbose=1, save_best_only=True,mode='auto',period=1)
    history = model.fit(X_train, y_train, epochs=20000, batch_size=128,class_weight=None,validation_data=(X_val, y_val), callbacks=[RocAuc,checkpoint],verbose=1)

def draw_history(history,title):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title+' Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title+' Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'val'], loc='upper left')
    plt.show()

df=pd.read_table('./Data.txt',sep='\t')
X_list=df.iloc[:,0:-1]
X=np.array(X_list)
y_list=df.iloc[:,-1]
y=np.array(y_list)
vals=np.array([])
y_vals=np.array([]).reshape(0,3)
y_val_scores=np.array([]).reshape(0,3)
y = to_categorical(y)
for train_index, test_index in skf1:
     print("TRAIN:", train_index, "TEST:", test_index)
     X_train, X_test = X[train_index], X[test_index]
     y_train, y_test = y[train_index], y[test_index]
     if i<=5 :
        i=i+1
        Train_fold(i, X_train, y_train, X_test, y_test)