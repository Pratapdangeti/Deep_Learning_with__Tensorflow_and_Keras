
# Timeseries model using LSTM in TensorFlow

import numpy as np
import pandas as pd
from numpy import expand_dims
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import tensorflow as tf
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

nse_data = pd.read_csv(data_path+'^NSEI.csv')

nse_data['Date'] = pd.to_datetime(nse_data['Date'], format='%Y-%m-%d', errors='ignore')
nse_data = nse_data[pd.notnull(nse_data['Close'])]

print(nse_data.head())

close_price_data = nse_data['Close']
close_price_data_intr = close_price_data.as_matrix()
close_price_data = expand_dims(close_price_data_intr, axis=1)


# Scaling continuous variables
scaler = MinMaxScaler(feature_range=(0,1))
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
    y_data = np.zeros((input_data.shape[0]-(lag+hist-1),1))
    for i in range((lag+hist-1),(input_data.shape[0]),1):
        y_data[i-(lag+hist-1),0] = input_data[i, 0]
        x_data_inter = input_data[(i-(hist+lag-1)):(i-(lag-1)), 0]
        x_data[i-(lag+hist-1),:] = x_data_inter.T
    return x_data,y_data

data_x_val, data_y_val = data_prep_lstm(close_price_data_2,hist=n_inputs,lag=lag_value)

# Function to split the data
def split_data(_x_data, _y_data,num_perd,test_split_ratio,_xdim, _ydim):
    num_rows = len(_x_data)
    test_split = num_perd *test_split_ratio

    # take data from end
    test_x_intr = _x_data[-test_split:]
    test_y_intr = _y_data[-test_split:]
    test_x = test_x_intr.reshape(-1,num_perd,_xdim)
    test_y = test_y_intr.reshape(-1,num_perd,_ydim)

    remain_x = _x_data[:(num_rows-test_split)]
    remain_y = _y_data[:(num_rows-test_split)]

    train_x_intr = remain_x[(len(remain_x)%num_perd):]
    train_y_intr = remain_y[(len(remain_y)%num_perd):]
    train_x = train_x_intr.reshape(-1,num_perd,_xdim)
    train_y = train_y_intr.reshape(-1,num_perd,_ydim)
    return train_x,test_x,train_y,test_y

x_train, x_test, y_train, y_test = split_data(data_x_val,data_y_val,num_perd =n_steps ,
                                              test_split_ratio=test_split_ratio_val,
                                              _xdim =n_inputs,_ydim = n_outputs)


x = tf.placeholder(tf.float32,[None, n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None, n_steps,n_outputs])


def LSTM_Model_TF(_x,_n_neurons,_n_outputs):
    # single layer of LSTM cell
    cell = tf.nn.rnn_cell.LSTMCell(_n_neurons,activation=tf.nn.relu)
    # dynamically unroll the network
    rnn_output, states = tf.nn.dynamic_rnn(cell,_x,dtype=tf.float32)
    stacked_rnn_output = tf.reshape(rnn_output,[-1,_n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_output,_n_outputs)
    outputs = tf.reshape(stacked_outputs,[-1,n_steps,_n_outputs])
    return outputs


def LSTM_Multilayer_Model_TF(_x,_n_neurons,_n_outputs):
    # 2 layers of LSTM cells
    num_units = 2*[_n_neurons]
    cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in num_units]
    # wrapping multiple layers into one Multi cell
    stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # to unroll the network dynamically
    rnn_output, states = tf.nn.dynamic_rnn(stacked_rnn_cell,_x,dtype=tf.float32)
    stacked_rnn_output = tf.reshape(rnn_output,[-1,_n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_output,_n_outputs)
    outputs = tf.reshape(stacked_outputs,[-1,n_steps,_n_outputs])
    return outputs


# Construct model
# Single layer LSTM model
output = LSTM_Model_TF(x,n_neurons,n_outputs)

# Multi layer LSTM model
#output = LSTM_Multilayer_Model_TF(x,n_neurons,n_outputs)


# Training model
cost_op = tf.reduce_mean(tf.square(output-y))
train_op =tf.train.AdamOptimizer(0.001).minimize(cost_op)

# number of epochs
n_epochs = 1000

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/tmp/tf_timeserieslogs", sess.graph)

    for ep in range(n_epochs):
        sess.run(train_op,feed_dict={x:x_train,y:y_train})
        if ep%10 == 0:
            train_mse = cost_op.eval(feed_dict = {x:x_train,y:y_train})
            test_mse = cost_op.eval(feed_dict={x: x_test, y: y_test})

            tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train mse", simple_value=train_mse)])
            writer.add_summary(tr_summary, global_step=ep)
            tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test mse", simple_value=test_mse)])
            writer.add_summary(tst_summary, global_step=ep)

            print(ep,"Train MSE :",train_mse,"Test MSE :",test_mse)

    y_pred_train = sess.run(output, feed_dict={x: x_train})
    y_pred_test = sess.run(output, feed_dict={x: x_test})

    # Converting back to original scale of stock price
    y_train_orig_scale = expand_dims(y_train.flatten(), axis=1)
    y_train_orig_scale = scaler.inverse_transform(y_train_orig_scale)

    y_pred_train_orig_scale = expand_dims(y_pred_train.flatten(), axis=1)
    y_pred_train_orig_scale = scaler.inverse_transform(y_pred_train_orig_scale)

    y_test_orig_scale = expand_dims(y_test.flatten(), axis=1)
    y_test_orig_scale = scaler.inverse_transform(y_test_orig_scale)

    y_pred_test_orig_scale = expand_dims(y_pred_test.flatten(), axis=1)
    y_pred_test_orig_scale = scaler.inverse_transform(y_pred_test_orig_scale)

    # Plotting the actual vs. predicted test signal
    plt.plot(y_test_orig_scale, 'k', label="test_actual")
    plt.title("Actual vs. Predicted of Closing Price TensorFlow ")
    plt.plot(y_pred_test_orig_scale,'r--',label="test_predicted")
    plt.legend(loc='upper left')
    plt.show()

    # Calculating r-square value
    print("Timeseries Train R-squared TensorFlow: ", r2_score(y_train_orig_scale, y_pred_train_orig_scale ))
    print("Timeseries Test R-squared TensorFlow: ", r2_score(y_test_orig_scale, y_pred_test_orig_scale))


