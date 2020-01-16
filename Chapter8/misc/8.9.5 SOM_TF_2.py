

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from keras.datasets import mnist



class SOM():
    def __init__(self):
        self.m = 20
        self.n = 30
        self.iterations = 400
        self.alpha = 0.3
        self.sigma = max(self.m,self.n)/2.0
        self.dim = 784

        # Dimensional weights of SOM
        self.weightage_vects = tf.Variable(tf.random_normal(shape=[m*n,dim],stddev=0.05))

    #Yields one by one the 2-D locations of the individual neurons
    def neuron_locations(self):
        for i in range(self.m):
            for j in range(self.n):
                yield np.array([i, j])


    location_vects = tf.constant(np.array(list(neuron_locations(self))))
        #vect_input = tf.placeholder("float",[dim])


def feedforward(_input):
    input_vecs = tf.expand_dims(tf.expand_dims(_input, 0),0)
    input_vecs = tf.cast(input_vecs, tf.float32)
    weight_vecs = tf.expand_dims(weightage_vects,0)
    grad_pass = tf.pow(tf.subtract(weight_vecs,input_vecs),2)
    squared_distance = tf.reduce_sum(grad_pass, 2)
    bmu_indices = tf.argmin(squared_distance, axis=1)
    bmu_locs = tf.reshape(tf.gather(location_vects, bmu_indices), [-1, 2])
    return bmu_locs


def backprop(_iter,_num_epoch,_alpha):

    radius = tf.subtract(sigma,tf.multiply(_iter,tf.divide(tf.cast(tf.subtract(_alpha,1),tf.float32),
                                                           tf.cast(tf.subtract(_num_epoch,1),tf.float32))))

    _alpha = tf.subtract(_alpha,tf.multiply(_iter,tf.divide(tf.cast(tf.subtract(_alpha,1),tf.float32),
                                                           tf.cast(tf.subtract(_num_epoch,1),tf.float32))))




    return _alpha



# Importing MNIST data
(x_train,y_train),(_,_) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],784)

x_train_f = x_train[0]

output = feedforward(x_train_f)

print(output)

op_backprop = backprop(_iter=1,_num_epoch=20,_alpha=alpha)

print(op_backprop)







print("Completed!")
