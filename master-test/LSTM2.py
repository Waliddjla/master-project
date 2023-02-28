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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
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
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import *
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


datatrain=train_dataset
datatest = test_dataset
data =dff
def create_dataset(data, look_back=8):
	dataX, dataY = [], []
	for i in range(len(data)-look_back):
		a = data[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(data[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

trainx,trainy =X_train, y_train
testx,testy = X_test ,y_test
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler_train.fit_transform(y_train)

scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(y_test)

look_back = 8
trainx,trainy = create_dataset(scaled_train, look_back)
testx, testy = create_dataset(scaled_test, look_back)
print("trainX: ", trainx.shape)
print("trainY: ", trainy.shape)
print("testX: ", testx.shape)
print("testY", testy.shape)
print("trainX: ", trainx)
# reshape input to be [samples, time steps, features]
trainx= np.reshape(trainx, (trainx.shape[0], trainx.shape[1], 1))

testx = np.reshape(testx, (testx.shape[0], testx.shape[1], 1 ))
print("trainX: ", trainx)
trainx.shape
testx.shape
trainy.shape
print("trainX.shape[1] - i.e. timesteps in input_shape = (timesteps, n_features) ", trainx.shape[1])
print("trainX.shape[2] - i.e. n_features in input_shape = (timesteps, n_features) ", trainx.shape[2])
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 

model1 = Sequential()
model1.add(LSTM(units = 256, activation = 'relu',return_sequences=True, input_shape = (trainx.shape[1],1)))
model1.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model1.add(LSTM(units = 128, input_shape = (trainx.shape[1], 1)))
#model1.add(LSTM(units = 128, return_sequences = True, input_shape = (trainx.shape[1], trainx.shape[2])))
model1.add(Dropout(0.2))
# Adding the output layer
model1.add(Dense(units = 1))
model1.summary()
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Compiling the LSTM
model1.compile(optimizer= 'adam', loss = 'mean_squared_error')

checkpoint_path = 'my_best_model.hdf5'

checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

earlystopping = EarlyStopping(monitor='loss', restore_best_weights=True)
#callbacks = [checkpoint, earlystopping]
callbacks = [checkpoint]
history = model1.fit(trainx, trainy, batch_size = 64, epochs =1000,validation_data=(testx, testy), callbacks=callbacks)
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 

model_from_saved_checkpoint = load_model(checkpoint_path)
plt.figure(figsize=(16,7))
plt.plot(model1.history['loss'], label='train')

plt.plot(model1.history['val_loss'], label='test')
plt.legend()
plt.show()

predicted = model_from_saved_checkpoint.predict(testx)
predicted = scaler_test.inverse_transform(predicted.reshape(-1, 1))
test_actual = scaler_test.inverse_transform(testy.reshape(-1, 1))
print('MAE :' , mean_absolute_error(test_actual,predicted))
print('RMSE:', np.sqrt(mean_squared_error(test_actual,predicted)))
print(r2_score(test_actual,predicted))

plt.figure(figsize=(16,7))
plt.plot(test_actual, marker='.', label='Actual Test')
plt.plot(predicted, 'r', marker='.', label='Predicted Test')
plt.legend()
plt.show()

predicted2 = model_from_saved_checkpoint.predict(trainx)

predicted2 = scaler_train.inverse_transform(predicted2.reshape(-1, 1))

train_actual = scaler_train.inverse_transform(trainy.reshape(-1, 1))
print('MAE :' , mean_absolute_error(train_actual,predicted2))
print('RMSE:', np.sqrt(mean_squared_error(train_actual,predicted2)))
print(r2_score(train_actual,predicted2))

plt.figure(figsize=(16,7))
plt.plot(train_actual, marker='.', label='Actual Train')
plt.plot(predicted2, 'r', marker='.', label='Predicted Train')
plt.legend()
plt.show()

a = plt.axes(aspect='equal')
plt.scatter(test_actual,predicted)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions6 ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)



testx.shape

lookback_period = 100

testX_last_days = testx[testx.shape[0] + lookback_period :  ]

testX_last_days.shape
predicted_days = []
for i in range(100):  
  predicted_forecast_test_x = model_from_saved_checkpoint.predict(testX_last_days[i:i+1])
  
  predicted_forecast_test_x = scaler_test.inverse_transform(predicted_forecast_test_x.reshape(-1, 1))
  # print(predicted_forecast_price_test_x)
  predicted_days.append(predicted_forecast_test_x)
  
print("Forecast for the next 100 Days Beyond the actual trading days ", np.array(predicted_days)) 
predicted_days = np.array(predicted_days)

predicted_days.shape
predicted.shape
predicted_days = predicted_days.flatten()
predicted_days
predicted = predicted.flatten()
predicted
predicted_concatenated = np.concatenate((predicted, predicted_days))

predicted_concatenated

predicted_concatenated.shape

plt.figure(figsize=(16,7))
plt.plot(test_actual, marker='.', label='Actual Test')
plt.plot(predicted_concatenated, 'r', marker='.', label='Predicted Test')

plt.legend()

plt.show()
