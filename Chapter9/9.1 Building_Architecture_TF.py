
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

print("X shape:",X.shape)
print("Y shape:",y.shape)

x_check = X[0:2,:]


# One-hot encoding the classes
num_classes = 10
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
y_check = y_[0:2,:]

# layers
input_dim = 64
layer_1_neurons = 30
layer_2_neurons = 20

# Build layer by layer until output shape do appear
W1=tf.Variable(tf.random_uniform([input_dim,layer_1_neurons],dtype=tf.float64))
b1=tf.Variable(tf.zeros([layer_1_neurons],dtype=tf.float64))

W2=tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons],dtype=tf.float64))
b2=tf.Variable(tf.zeros([layer_2_neurons],dtype=tf.float64))

WO=tf.Variable(tf.random_uniform([layer_2_neurons,num_classes],dtype=tf.float64))
bo=tf.Variable(tf.zeros([num_classes],dtype=tf.float64))

# Layer 1
layer_1 = tf.add(tf.matmul(x_check,W1),b1)
layer_1 = tf.nn.relu(layer_1)

# Layer 2
layer_2 = tf.add(tf.matmul(layer_1, W2), b2)
layer_2 = tf.nn.relu(layer_2)
# Output Layer
output_layer = tf.add(tf.matmul(layer_2,WO),bo)

# Checking shapes of all the variables across the flow
print("X check input shape:",x_check.shape)
print("Layer 1 shape:",layer_1.shape)
print("Layer 2 shape:",layer_2.shape)
print("Output shape:",output_layer.shape)
print("y check input shape:",y_check.shape)


# Now putting operations inside the function
def Multi_Class_TF(_x_in):
    # Layer 1
    _W1 = tf.Variable(tf.random_uniform([input_dim, layer_1_neurons], dtype=tf.float64))
    _b1 = tf.Variable(tf.zeros([layer_1_neurons], dtype=tf.float64))
    _layer_1 = tf.add(tf.matmul(_x_in,_W1),_b1)
    _layer_1 = tf.nn.relu(_layer_1)

    # Layer 2
    _W2 = tf.Variable(tf.random_uniform([layer_1_neurons, layer_2_neurons], dtype=tf.float64))
    _b2 = tf.Variable(tf.zeros([layer_2_neurons], dtype=tf.float64))
    _layer_2 = tf.add(tf.matmul(_layer_1,_W2),_b2)
    _layer_2 = tf.nn.relu(_layer_2)

    # Output Layer
    _WO = tf.Variable(tf.random_uniform([layer_2_neurons, num_classes], dtype=tf.float64))
    _bo = tf.Variable(tf.zeros([num_classes], dtype=tf.float64))
    _output_layer = tf.add(tf.matmul(_layer_2,_WO),_bo)

    return _output_layer


# Testing function on sample data
y_func_test = Multi_Class_TF(x_check)
print("Y func sample shape test:",y_func_test.shape)


# Testing function on whole data
y_func_full_test = Multi_Class_TF(X)
print("Y func full shape test:",y_func_full_test.shape)

# Now disable eager execution and create variables like placeholders
xs = tf.placeholder(tf.float64,[None,input_dim],name="Input_data")
ys = tf.placeholder(tf.float64,[None,num_classes],name="output_data")

# Construct model
output = Multi_Class_TF(xs)


print("Completed")

