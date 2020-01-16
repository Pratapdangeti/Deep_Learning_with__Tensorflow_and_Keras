"""
Author - Pratap Dangeti
"""

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
[0.00632,2.31,0.538,65.2],
[0.02731,7.07,0.469,78.9],
[0.02729,7.07,0.469,61.1],
[0.03237,2.18,0.458,45.8],
[0.06905,2.18,0.458,54.2]
],dtype= np.float64)

y= np.array([
[24],
[21.6],
[34.7],
[33.4],
[36.2]
])

# Layer architecture
inputlayer_neurons = X.shape[1]
hiddenlayer_1_neurons = 3
hiddenlayer_2_neurons = 3
output_neurons = 1
lr = 0.01


# Weight and bias initialization
np.random.seed(10)

wh1 = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_1_neurons))
bh1 = np.random.uniform(size=(1, hiddenlayer_1_neurons))

wh2 = np.random.uniform(size=(hiddenlayer_1_neurons, hiddenlayer_2_neurons))
bh2 = np.random.uniform(size=(1, hiddenlayer_2_neurons))

wout = np.random.uniform(size=(hiddenlayer_2_neurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# Activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivatives_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def linear(x):
    return x

def derivatives_linear(x):
    return np.ones(np.shape(x))



"""

# Forward Propagation
hidden_layer_1_input_bfr = np.dot(X, wh1)
hidden_layer_1_input = hidden_layer_1_input_bfr + bh1
hidden_layer_1_activation = sigmoid(hidden_layer_1_input)

hidden_layer_2_input_bfr = np.dot(hidden_layer_1_activation, wh2)
hidden_layer_2_input = hidden_layer_2_input_bfr + bh2
hidden_layer_2_activation = sigmoid(hidden_layer_2_input)

output_layer_input_bfr = np.dot(hidden_layer_2_activation, wout)
output_layer_input = output_layer_input_bfr + bout
output = linear(output_layer_input)

print("\n Output :\n",output)

error = output-y
print("\n Error :\n",error)

# Backpropagation
slope_output_layer = derivatives_linear(output)
slope_hidden_layer_2 = derivatives_sigmoid(hidden_layer_2_activation)
slope_hidden_layer_1 = derivatives_sigmoid(hidden_layer_1_activation)

d_output = error * slope_output_layer

error_at_hidden_layer_2 = np.dot(d_output,wout.T)
d_hiddenlayer_2 = error_at_hidden_layer_2 * slope_hidden_layer_2

error_at_hidden_layer_1 = np.dot(d_hiddenlayer_2,wh2.T)
d_hiddenlayer_1 = error_at_hidden_layer_1 * slope_hidden_layer_1

print("\n before wout :\n",wout)
wout -= np.dot(hidden_layer_2_activation.T,d_output)*lr
print("\n after wout :\n",wout)

print("\n before bout :\n",bout)
bout -= np.sum(d_output,axis=0,keepdims=True)*lr
print("\n after bout :\n",bout)

print("\n before wh2 :\n",wh2)
wh2 -= np.dot(hidden_layer_1_activation.T,d_hiddenlayer_2)*lr
print("\n after wh2 :\n",wh2)

print("\n before bh2 :\n",bh2)
bh2 -= np.sum(d_hiddenlayer_2,axis=0,keepdims=True)*lr
print("\n after bh2 :\n",bh2)

print("\n before wh1 :\n",wh1)
wh1 -= np.dot(X.T,d_hiddenlayer_1) *lr
print("\n after wh1 :\n", wh1)

print("\n before bh1 :\n",bh1)
bh1 -= np.sum(d_hiddenlayer_1,axis=0,keepdims=True)* lr
print("\n after bh1 :\n",bh1)

"""

# Function to calculate mse error for plot
def mean_squared_error(actual, predicted):
    mse_val = np.mean(np.square(actual-predicted))
    return mse_val

# Iterations
epoch = 100
# Error array for storing mse results at each epoch
error_array = np.zeros((epoch,2))
# Final output for printing
final_output = np.zeros((epoch,1))

for i in range(epoch):

    # Forward Propagation
    hidden_layer_1_input_bfr = np.dot(X, wh1)
    hidden_layer_1_input = hidden_layer_1_input_bfr + bh1
    hidden_layer_1_activation = sigmoid(hidden_layer_1_input)

    hidden_layer_2_input_bfr = np.dot(hidden_layer_1_activation, wh2)
    hidden_layer_2_input = hidden_layer_2_input_bfr + bh2
    hidden_layer_2_activation = sigmoid(hidden_layer_2_input)

    output_layer_input_bfr = np.dot(hidden_layer_2_activation, wout)
    output_layer_input = output_layer_input_bfr + bout
    output = linear(output_layer_input)
    error =  output - y

    mse_error = mean_squared_error(y, output)
    error_array[i,0]= int(i+1)
    error_array[i,1] = mse_error

    # Backpropagation
    slope_output_layer = derivatives_linear(output)
    slope_hidden_layer_2 = derivatives_sigmoid(hidden_layer_2_activation)
    slope_hidden_layer_1 = derivatives_sigmoid(hidden_layer_1_activation)

    d_output = error * slope_output_layer

    error_at_hidden_layer_2 = np.dot(d_output,wout.T)
    d_hiddenlayer_2 = error_at_hidden_layer_2 * slope_hidden_layer_2

    error_at_hidden_layer_1 = np.dot(d_hiddenlayer_2,wh2.T)
    d_hiddenlayer_1 = error_at_hidden_layer_1 * slope_hidden_layer_1

    wout -= np.dot(hidden_layer_2_activation.T,d_output)*lr
    bout -= np.sum(d_output,axis=0,keepdims=True)*lr
    wh2 -= np.dot(hidden_layer_1_activation.T,d_hiddenlayer_2)*lr
    bh2 -= np.sum(d_hiddenlayer_2,axis=0,keepdims=True)*lr
    wh1 -= np.dot(X.T,d_hiddenlayer_1) *lr
    bh1 -= np.sum(d_hiddenlayer_1,axis=0,keepdims=True)* lr

    if i == (epoch-1):
        final_output = output


print("\n Predicted Output :\n",final_output)


plt.plot(error_array[:,0],error_array[:,1])
#plt.ylim(0,100)
#plt.title("Errors of Forward and Backpropagation ")
plt.xlabel("Number of epochs")
plt.ylabel("Training loss")
plt.show()
plt.close()



print("completed")





