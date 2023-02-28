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
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import MeanSquaredLogarithmicError
import math
import streamlit as st
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score

dfn = pd.read_csv("data.csv")
print (dfn) 
df = dfn
df.drop(['Jul','Stn Id', 'Stn Name', 'CIMIS Region', 'qc','qc.1', 'qc.2', 'qc.3','qc.4','qc.5','qc.6','qc.7','qc.8','qc.9','qc.10','qc.11','qc.12','qc.13','qc.14','qc.15','qc.16','qc.17','qc.18','qc.19','qc.20','qc.21','qc.22','qc.23','qc.24','qc.25','qc.26','qc.27','qc.28','qc.29','Batt Volt (VDC)', ], axis = 1, inplace = True)
df.drop(['Rose ESE','Rose ESE', 'Rose SSE', 'Rose SSW', 'Rose WSW', 'Rose WNW','Rose NNW'], axis = 1, inplace = True)
df.drop(['Max Vap Pres (kPa)','Min Vap Pres (kPa)','Min Air Temp (C)', 'Max Air Temp (C)','Max Rel Hum (%)','Min Rel Hum (%)','Max Soil Temp (C)','Min Soil Temp (C)',], axis = 1, inplace = True)
df.drop(['Rose NNE', 'Rose ENE','Exp 1', 'Exp 2','Dew Point (C)','PM ETr (mm)'], axis = 1, inplace = True)
df.drop(['Precip (mm)','ETo (mm)'], axis=1,inplace = True)
df.dtypes

df.columns.values
df.info()
df = df.fillna({'Sol Rad (W/sq.m)': df['Sol Rad (W/sq.m)'].median()})
df = df.fillna({'Net Rad (W/sq.m)': df['Net Rad (W/sq.m)'].median()})
df = df.fillna({'Avg Air Temp (C)': df['Avg Air Temp (C)'].median()})
df = df.fillna({'Avg Rel Hum (%)': df['Avg Rel Hum (%)'].median()})
df = df.fillna({'Avg Wind Speed (m/s)': df['Avg Wind Speed (m/s)'].median()})
df = df.fillna({'Wind Run (km)': df['Wind Run (km)'].median()})
df = df.fillna({'Avg Soil Temp (C)': df['Avg Soil Temp (C)'].median()})
df = df.fillna({'PM ETo (mm)': df['PM ETo (mm)'].median()})
df = df.fillna({'Avg Vap Pres (kPa)': df['Avg Vap Pres (kPa)'].median()})
#df = df.fillna({'Precip (mm)': df['Precip (mm)'].median()})

df.isna().sum()
na = df.isna()
corr = df.corr()
df.isna().any().any()
des = df.describe()
print(corr)
sns.heatmap(corr)
sb.pairplot(df)
df.shape
df.head()
df.index
from pathlib import Path  
filepath = Path('C:/Users/Click/Desktop/pfe-walid/out.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
df.to_csv(filepath) 
df = df.set_index('Date')
df = df.set_index(pd.to_datetime(df.index, format='%m/%d/%Y').strftime("%Y-%m-%d"))
#x = df.drop(['ETo (mm)','Precip (mm)', 'PM ETo (mm)'],axis = 1)
#y = df['PM ETo (mm)']
dff= df
filepath = Path('C:/Users/Click/master-test/DFF.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
dff.to_csv(filepath) 

plt.figure(figsize=(15,12))
plt.suptitle('Lag Plots', fontsize=22)

plt.subplot(3,3,1)
pd.plotting.lag_plot(df['PM ETo (mm)'], lag=1) #Daily lag
plt.title('Daily Lag')

plt.subplot(3,3,2)
pd.plotting.lag_plot(df['PM ETo (mm)'], lag=7) #weekly lag
plt.title('Weekly Lag')

plt.subplot(3,3,3)
pd.plotting.lag_plot(df['PM ETo (mm)'], lag=30) #month lag
plt.title('1-Month Lag')
plt.subplot(3,3,4)
pd.plotting.lag_plot(df['PM ETo (mm)'], lag=365) #month lag
plt.title('1-year Lag')
plt.legend()
plt.show()
df.plot(figsize=(50,50))
plot_cols = ['Sol Rad (W/sq.m)', 'Net Rad (W/sq.m)', 'Avg Vap Pres (kPa)','PM ETo (mm)']
plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols]
_ = plot_features.plot(subplots=True)

plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


df.hist(figsize = (12,10))
plt.show()
k= df.head()
#X_trainx, X_testx, y_trainy, y_testy = train_test_split(x, y, test_size=0.2,random_state=42 )
# Split test data to X and y
train_size = int(len(df)*0.8)
print(train_size)
train_dataset, test_dataset = df.iloc[:train_size], df.iloc[train_size:]

X_train = train_dataset.drop(['PM ETo (mm)'], axis=1)
y_train = train_dataset.loc[:, ['PM ETo (mm)']]

X_test = test_dataset.drop(['PM ETo (mm)'], axis=1)
y_test = test_dataset.loc[:, ['PM ETo (mm)']]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print (y_test.shape)
df.columns
X_train.columns
aim='PM ETo (mm)'
def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('ETo', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
line_plot(train_dataset[aim], test_dataset[aim], 'training', 'test', title='')

#LR
reg = LinearRegression()
reg.fit(X_train, y_train)
print( reg.coef_)
prediction = reg.predict(X_test)
print (reg.score(X_test,y_test))
reg.score(X_test, y_test )

print('MAE :' , mean_absolute_error(y_test,prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction)))
           
plt.figure(figsize = (24,7))
plt.plot(y_test)
plt.plot(prediction)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'prediction'], loc='upper right')
a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


#SVR
reg1 = SVR(kernel='rbf')
reg1.fit(X_train, y_train)
print (reg1.score(X_train, y_train))
print (reg1.score(X_test, y_test))
prediction1 = reg1.predict(X_test)
print(reg1.intercept_, reg1.coef_)

plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(prediction1)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'prediction1'], loc='upper right')
a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction1)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions1 ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
#kNR
knnr = KNeighborsRegressor()
knnr.fit(X_train,y_train)
prediction2 = knnr.predict(X_test)
knnr.score(X_test, y_test)
error = 1 - knnr.score(X_test, y_test)

a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction2)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(prediction2)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'prediction2'], loc='upper right')
#lasso
reglasso = Lasso(alpha=1.0)
reglasso.fit(X_train,y_train)
predlasso= reglasso.predict(X_test)
print(reglasso.score(X_test, y_test))
a = plt.axes(aspect='equal')
plt.scatter(X_test,predlasso)
plt.xlabel('True Values ETo')
plt.ylabel('Predlasso ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(predlasso)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'predlasso'], loc='upper right')
#XGBR
XGBR = XGBRegressor(n_estimators=500)
XGBR.fit(X_train , y_train)
print(XGBR.score(X_train, y_train))
prediction5= XGBR.predict(X_test)
a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction5)
plt.xlabel('True Values ETo')
plt.ylabel('Prediction5 ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(prediction5)
plt.xlabel('Time')
plt.ylabel('pred5 ETO')
plt.legend(['Train set', 'predxgbr'], loc='upper right')
#RANDOMFOREST
RandomForestRegModel = RandomForestRegressor(n_estimators=500)
RandomForestRegModel.fit(X_train , y_train)
print(RandomForestRegModel.score(X_train, y_train))
prediction6= RandomForestRegModel.predict(X_test) 
a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction6)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions6 ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(prediction6)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'predforest'], loc='upper right')
#decisiontree
regtree = DecisionTreeRegressor(max_depth=8)
regtree.fit(X_train, y_train)
predtree = regtree.predict(X_test)
print (regtree.score(X_test,y_test))
a = plt.axes(aspect='equal')
plt.scatter(y_test,predtree)
plt.xlabel('True Values ETo')
plt.ylabel('Predtree ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(predtree)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'predtree'], loc='upper right')

#MLP
lbfgs = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 100))
lbfgs.fit(X_train , y_train)
predlbgf=lbfgs.predict(X_train)
print(lbfgs.score(X_train, y_train))
a = plt.axes(aspect='equal')
plt.scatter(y_test,predlbgf)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

plt.plot(figsize=(30,30))
plt.plot(prediction2, "gd", label="KNeighborsRegressor")
plt.plot(prediction1, "b^", label="SVR")
plt.plot(prediction, "ys", label="LinearRegression")
plt.plot(predlasso, "r*", ms=8, label="LassoRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")
plt.show()


from sklearn.tree import plot_tree
plt.figure(figsize=(20,20),dpi=200)
plot_tree(regtree, feature_names=X_test.columns)

a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction5)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.xticks(())
plt.yticks(())

plt.show()
import math
from sklearn.metrics import mean_squared_error,accuracy_score

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
NNpredictio1 = NN_model.predict(X_train)
print('MAE :' , mean_absolute_error(y_test,NNprediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test,NNprediction)))
plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(NNprediction)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'prediction'], loc='upper right')

a = plt.axes(aspect='equal')
plt.scatter(y_test,NNprediction)
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
print(r2_score(y_test,NNprediction))
print('MAE :' , mean_absolute_error(testy,predicted))
print('RMSE:', np.sqrt(mean_squared_error(testy,predicted))) 

print(cross_val_score(XGBR, testy, predicted))
print(r2_score(testy, predicted))


# reshape input to be [samples, time steps, features]

datatrain=train_dataset
datatest = test_dataset
data =df
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
model1 = Sequential()
model1.add(LSTM(units = 256, activation = 'relu',return_sequences=True, input_shape = (trainx.shape[1], 1)))
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
model1.compile(optimizer = 'adam', loss = 'mean_squared_error')

checkpoint_path = 'my_best_model.hdf5'

checkpoint = ModelCheckpoint(filepath=checkpoint_path, 
                             monitor='loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')

#earlystopping = EarlyStopping(monitor='loss',  patience= 200, restore_best_weights=True)
#callbacks = [checkpoint, earlystopping]
callbacks = [checkpoint]
model1.fit(trainx, trainy, batch_size = 128, verbose=1, epochs = 700,validation_data=(testx, testy), callbacks=callbacks)
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 

model_from_saved_checkpoint = load_model(checkpoint_path)

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

testX_last_days = testx[testx.shape[0] - lookback_period :  ]

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




