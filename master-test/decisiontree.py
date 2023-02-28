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
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import math
from sklearn.metrics import *
from sklearn.metrics import r2_score
import time

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

start_time = time.time()
regtree = DecisionTreeRegressor(max_depth=9)
regtree.fit(X_train, y_train)
prediction = regtree.predict(X_test)
print (regtree.score(X_test,y_test))
print("--- %s seconds ---" %(time.time() -  start_time))
print('MAE :' , mean_absolute_error(y_test,prediction))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction))) 
print(r2_score(y_test,prediction))

x_ax = range(len(y_test))
plt.plot(x_ax, y_test.values, label="original")
plt.plot(x_ax, prediction, label="predicted")
plt.title("ETo test and DecisionTree predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()           


a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction, color='red')
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
_ = plt.plot(lims, lims)



plt.figure(figsize = (20,10))
plt.plot(y_test)
plt.plot(prediction)
plt.xlabel('Time')
plt.ylabel('ETO')
plt.legend(['Train set', 'predtree'], loc='upper right')

x_ax = range(len(y_test))
plt.plot(x_ax, y_test.values, label="original")
plt.plot(x_ax, prediction, label="predicted")
plt.title("ETo test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
# one parmetre
start_time = time.time() 
regg1 = DecisionTreeRegressor(max_depth=5)
regg1.fit(X_train[['Avg Rel Hum (%)']], y_train)
prediction1 = regg1.predict(X_test[['Avg Rel Hum (%)']])
print (regg1.score(X_test[['Avg Rel Hum (%)']],y_test))
print("--- %s seconds ---" %(time.time() -  start_time))
print('MAE :' , mean_absolute_error(y_test,prediction1))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction1)))
print(r2_score(y_test, prediction1))
#with 2
start_time = time.time()
regg2 = DecisionTreeRegressor(max_depth=8)
regg2.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)']], y_train)
prediction2 = regg2.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)']])
print (regg2.score(X_test[['Avg Rel Hum (%)','Wind Run (km)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('MAE :' , mean_absolute_error(y_test,prediction2))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction2)))
print(r2_score(y_test, prediction2))
#with 3
start_time = time.time()
regg3 = DecisionTreeRegressor(max_depth=5)
regg3.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)']], y_train)
prediction3 = regg3.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)']])
print (regg3.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('MAE :' , mean_absolute_error(y_test,prediction3))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction3)))
print(r2_score(y_test, prediction3))
#with 4
start_time = time.time()
regg4 =DecisionTreeRegressor(max_depth=8)
regg4.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)']], y_train)
prediction4 = regg4.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)']])
print (regg4.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('MAE :' , mean_absolute_error(y_test,prediction4))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction4)))
print('score:',r2_score(y_test,prediction4))
#with 5
start_time = time.time()
regg5 = DecisionTreeRegressor(max_depth=6)
regg5.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)']], y_train)
prediction5 = regg5.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)']])
print (regg5.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('MAE :' , mean_absolute_error(y_test,prediction5))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction5)))
print('score:',r2_score(y_test,prediction5))
#♣with 6
start_time = time.time()
regg6 = DecisionTreeRegressor(max_depth=7)
regg6.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)']], y_train)
prediction6 = regg6.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)']])
print (regg6.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('score:',r2_score(y_test,prediction6))
print('MAE :' , mean_absolute_error(y_test,prediction6))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction6))) 
#with 7
start_time = time.time()
regg7 = DecisionTreeRegressor(max_depth=8)
regg7.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)']], y_train)
prediction7 = regg7.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)']])
print (regg7.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Net Rad (W/sq.m)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('score:',r2_score(y_test,prediction7))
print('MAE :' , mean_absolute_error(y_test,prediction7))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction7)))
#with the first corr
start_time = time.time()
regg8 = DecisionTreeRegressor(max_depth=8)
regg8.fit(X_train[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Sol Rad (W/sq.m)']], y_train)
prediction8 = regg8.predict(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Sol Rad (W/sq.m)']])
print (regg8.score(X_test[['Avg Rel Hum (%)','Wind Run (km)','Avg Wind Speed (m/s)', 'Avg Vap Pres (kPa)','Avg Air Temp (C)','Avg Soil Temp (C)','Sol Rad (W/sq.m)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('score:',r2_score(y_test,prediction8))
print('MAE :' , mean_absolute_error(y_test,prediction8))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction8)))
#with the best corr
start_time = time.time()
regg9 = DecisionTreeRegressor(max_depth=3)
regg9.fit(X_train[['Net Rad (W/sq.m)','Sol Rad (W/sq.m)']], y_train)
prediction9 = regg9.predict(X_test[['Net Rad (W/sq.m)','Sol Rad (W/sq.m)']])
print (regg9.score(X_test[['Net Rad (W/sq.m)','Sol Rad (W/sq.m)']],y_test))
print("--- %s seconds ---" %(time.time() - start_time))
print('score:',r2_score(y_test,prediction9))
print('MAE :' , mean_absolute_error(y_test,prediction9))
print('RMSE:', np.sqrt(mean_squared_error(y_test,prediction9)))
#plot
x_ax = range(len(y_test))
plt.plot(x_ax, y_test.values, label="original")
plt.plot(x_ax, prediction9, label="predicted")
plt.title("ETo test and RandomForest predicted data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()           


a = plt.axes(aspect='equal')
plt.scatter(y_test,prediction9, color='red')
plt.xlabel('True Values ETo')
plt.ylabel('Predictions ETo')
lims = [0,8]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
_ = plt.plot(lims, lims)