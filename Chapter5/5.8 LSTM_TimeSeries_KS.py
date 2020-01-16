

# Timeseries model using LSTM in Keras
import numpy as np
import pandas as pd
from numpy import expand_dims
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

from keras.layers import Dense,LSTM
from keras.models import Sequential
from keras.optimizers import Adam


pd.set_option('display.max_columns', 10)

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

nse_data = pd.read_csv(data_path+'^NSEI.csv')
nse_data['Date'] = pd.to_datetime(nse_data['Date'], format='%Y-%m-%d', errors='ignore')
nse_data = nse_data[pd.notnull(nse_data['Close'])]
close_price_data = nse_data['Close']
close_price_data_intr = close_price_data.as_matrix()
close_price_data = expand_dims(close_price_data_intr, axis=1)


# Scaling continuous variables
scaler = MinMaxScaler()
close_price_data_2 = scaler.fit_transform(close_price_data)

# Parameters
n_inputs = 1
n_outputs = 1
lag_value = 1
n_neurons = 10
n_steps = 4
test_split_ratio_val = 20


# Function to create number of history points and lags to predict variable
def data_prep_lstm(input_data,hist = 1,lag=1):
    x_data = np.zeros((input_data.shape[0]-(lag+hist-1),hist))
    y_data = np.zeros(input_data.shape[0]-(lag+hist-1))
    for i in range((lag+hist-1),(input_data.shape[0]),1):
        y_data[i-(lag+hist-1)] = input_data[i, 0]
        x_data_inter = input_data[(i-(hist+lag-1)):(i-(lag-1)), 0]
        x_data[i-(lag+hist-1),:] = x_data_inter.T
    return x_data,y_data

data_x_val, data_y_val = data_prep_lstm(close_price_data_2,hist=n_steps,lag=lag_value)
data_x_val = data_x_val.reshape((data_x_val.shape[0], data_x_val.shape[1], n_inputs))

# Function to split the data
def split_data(_x_data, _y_data,num_perd,test_split_ratio,_xdim, _ydim):
    num_rows = len(_x_data)
    test_split = num_perd *test_split_ratio

    # take data from end
    test_x = _x_data[-test_split:]
    test_y = _y_data[-test_split:]

    remain_x = _x_data[:(num_rows-test_split)]
    remain_y = _y_data[:(num_rows-test_split)]

    train_x = remain_x[(len(remain_x)%num_perd):]
    train_y = remain_y[(len(remain_y)%num_perd):]
    return train_x,test_x,train_y,test_y

x_train, x_test, y_train, y_test = split_data(data_x_val,data_y_val,num_perd =n_steps ,
                                              test_split_ratio=test_split_ratio_val,
                                              _xdim =n_inputs,_ydim = n_outputs)


def LSTM_Model_KS(_n_neurons,_n_steps,_n_inputs):
    model = Sequential()
    model.add(LSTM(_n_neurons,activation='relu',input_shape=(_n_steps,_n_inputs)))
    model.add(Dense(1))
    adam_opt = Adam(lr=0.01)
    # Model compilation
    model.compile(loss='mse',optimizer=adam_opt)
    return model



def LSTM_Multilayer_Model_KS(_n_neurons,_n_steps,_n_inputs):
    model=Sequential()
    model.add(LSTM(_n_neurons,activation='relu',
                   input_shape=(_n_steps,_n_inputs),
                   return_sequences=True))
    model.add(LSTM(_n_neurons,activation='relu',
                   return_sequences=False))
    model.add(Dense(1))
    adam_opt = Adam(lr=0.001)
    # Model compilation
    model.compile(loss='mse',optimizer=adam_opt)
    return model


# Model training
training_epochs = 1000
batch_size = 30

timeseries_model = LSTM_Model_KS(n_neurons,n_steps,n_inputs)
#timeseries_model = LSTM_Multilayer_Model_KS(n_neurons,n_steps,n_inputs)

timeseries_model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs)

y_pred_train = timeseries_model.predict(x_train)
y_pred_test = timeseries_model.predict(x_test)

# Converting back to original scale of stock price
y_train_2 = expand_dims(y_train, axis=1)
y_train_orig_scale = scaler.inverse_transform(y_train_2)
y_test_2 = expand_dims(y_test, axis=1)
y_test_orig_scale = scaler.inverse_transform(y_test_2)

y_pred_train_orig_scale = scaler.inverse_transform(y_pred_train)
y_pred_test_orig_scale = scaler.inverse_transform(y_pred_test)


plt.plot(y_test_orig_scale, 'k', label="test_actual")
plt.title("Actual vs. Predicted of Closing Price Keras ")
plt.plot(y_pred_test_orig_scale, 'r--', label="test_predicted")
plt.legend(loc='upper left')
plt.show()


print("Timeseries Train R-squared Keras : ", r2_score(y_train_orig_scale, y_pred_train_orig_scale))
print("Timeseries Test R-squared Keras : ", r2_score(y_test_orig_scale, y_pred_test_orig_scale ))


