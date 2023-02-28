import seaborn as sns
import seaborn as sb
import numpy as np
import tensorflow as tf
import tensorflow as keras
import matplotlib.pyplot as plt 
import datetime as dt
from datetime import datetime
from matplotlib import dates as mdates 
import pandas as pd
import matplotlib_inline 
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from keras.models import load_model
from keras.layers import LSTM,  Embedding
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.multioutput import MultiOutputRegressor 
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import math
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
dff = pd.read_csv("DFF.csv")
dff = dff.set_index('Date')
train_size = int(len(dff)*0.8)
print(train_size)
train_dataset, test_dataset = dff.iloc[:train_size], dff.iloc[train_size:]

X_train = train_dataset.drop(['PM ETo (mm)'], axis=1)
y_train = train_dataset.loc[:, ['PM ETo (mm)']]

X_test = test_dataset.drop(['PM ETo (mm)'], axis=1)
y_test = test_dataset.loc[:, ['PM ETo (mm)']]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print (y_test.shape)
dff.columns
X_train.columns
# with 1
X_train1 = train_dataset.drop(['PM ETo (mm)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test1 = test_dataset.drop(['PM ETo (mm)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
#with 2
X_train2 = train_dataset.drop(['PM ETo (mm)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test2 = test_dataset.drop(['PM ETo (mm)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
#with 3
X_train3 = train_dataset.drop(['PM ETo (mm)','Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test3 = test_dataset.drop(['PM ETo (mm)','Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
# with 4
X_train4 = train_dataset.drop(['PM ETo (mm)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test4 = test_dataset.drop(['PM ETo (mm)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
#with5
X_train5 = train_dataset.drop(['PM ETo (mm)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test5 = test_dataset.drop(['PM ETo (mm)','Avg Soil Temp (C)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
#with 6
X_train6 = train_dataset.drop(['PM ETo (mm)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
X_test6 = test_dataset.drop(['PM ETo (mm)','Net Rad (W/sq.m)','Sol Rad (W/sq.m)'], axis=1)
#with7
X_train7 = train_dataset.drop(['PM ETo (mm)','Sol Rad (W/sq.m)'], axis=1)
X_test7 = test_dataset.drop(['PM ETo (mm)','Sol Rad (W/sq.m)'], axis=1)
#with7.2
X_train8 = train_dataset.drop(['PM ETo (mm)','Net Rad (W/sq.m)',], axis=1)
X_test8 = test_dataset.drop(['PM ETo (mm)','Net Rad (W/sq.m)',], axis=1)
#plus corr2
X_train9 = train_dataset.drop(['PM ETo (mm)','Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)'], axis=1)
X_test9 = test_dataset.drop(['PM ETo (mm)','Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)'], axis=1)
#plus corr3 
X_train10 = train_dataset.drop(['PM ETo (mm)','Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)'], axis=1)
X_test10 = test_dataset.drop(['PM ETo (mm)','Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)'], axis=1)

NN_model = Sequential()
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()
#checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
#checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
#callbacks_list = [checkpoint]
NN_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split = 0.2)
NNprediction = NN_model.predict(X_test)
NNprediction1 = NN_model.predict(X_train)
print('MAE :' , mean_absolute_error(y_test,NNprediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test,NNprediction)))
print(r2_score(y_test,NNprediction))

plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(NNprediction)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'prediction'], loc='upper right')

x_ax = range(len(y_test))
plt.plot(x_ax, y_test.values, label="original")
plt.plot(x_ax, NNprediction, label="predicted")
plt.title("ETo test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(y_test,NNprediction)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

