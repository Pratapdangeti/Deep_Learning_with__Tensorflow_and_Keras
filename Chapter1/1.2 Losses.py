

import numpy as np

y_actual_reg = np.array([[24],[21.6],[34.7],[33.4],[36.2]])
y_pred_reg = np.array([[23.1],[22.3],[32.2],[33.1],[37.8]])

def mean_squared_error(actual, predicted):
    mse_val = np.mean(np.square(actual-predicted))
    return mse_val

print("mean square error :",mean_squared_error(y_actual_reg,y_pred_reg))


# Binary cross entropy
y_act_bin_array = np.array([[1],[0],[0],[1],[1],[0]])
y_pred_bin_array = np.array([[0.91],[0.26],[0.13],[0.84],[0.42],[0.79]])

def binary_cross_entropy(actual,predicted):
    mean_loss = -np.mean(actual*np.log(predicted + 1e-15))
    return mean_loss

print("binary entropy :",binary_cross_entropy(y_act_bin_array,y_pred_bin_array))


# Categorical cross entropy
y_act_cat_array = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,0,0],
    [0,0,1]
])

y_pred_cat_array = np.array([
    [0.8,0.1,0.1],
    [0.2,0.7,0.1],
    [0.3,0.3,0.4],
    [0.6,0.3,0.1],
    [0.3,0.1,0.6]
])


def cat_cross_entropy(actual,predicted):
    total_loss = (np.multiply(actual,np.log(predicted + 1e-15))).sum()
    mean_loss = total_loss/np.shape(actual)[0]
    return -mean_loss


print("categorical cross entropy :",cat_cross_entropy(y_act_cat_array,y_pred_cat_array))












