
import tensorflow as tf
import numpy as np


class SOM_Layer():
    def __init__(self ,m ,n ,dim ,num_epoch ,learning_rate_som ,radius_factor, gaussian_std):
        self.m = m
        self.n = n
        self.dim = dim
        self.gaussian_std = gaussian_std
        self.num_epoch = num_epoch
        self.map = tf.Variable(tf.random_normal(shape=[m * n, dim], stddev=0.05))
        self.location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
        self.alpha = learning_rate_som
        self.sigma = max(m, n) * radius_factor

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons in the SOM.
        """
        # Nested iterations over both dimensions to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def getmap(self):
        return self.map

    def getlocation(self):
        return self.bmu_locs

    def feedforward(self, input):
        self.input = input
        self.grad_pass = tf.pow(tf.subtract(tf.expand_dims(self.map, axis=0), tf.expand_dims(self.input, axis=1)), 2)
        self.squared_distance = tf.reduce_sum(self.grad_pass, 2)
        self.bmu_indices = tf.argmin(self.squared_distance, axis=1)
        self.bmu_locs = tf.reshape(tf.gather(self.location_vects, self.bmu_indices), [-1, 2])

    def backprop(self, iter, num_epoch):

        # Update the weigths
        radius = tf.subtract(self.sigma,
                             tf.multiply(iter,
                                         tf.divide(tf.cast(tf.subtract(self.alpha, 1), tf.float32),
                                                   tf.cast(tf.subtract(num_epoch, 1), tf.float32))))

        alpha = tf.subtract(self.alpha,
                            tf.multiply(iter,
                                        tf.divide(tf.cast(tf.subtract(self.alpha, 1), tf.float32),
                                                  tf.cast(tf.subtract(num_epoch, 1), tf.float32))))

        self.bmu_distance_squares = tf.reduce_sum(
            tf.pow(tf.subtract(tf.expand_dims(self.location_vects, axis=0),
                tf.expand_dims(self.bmu_locs, axis=1)), 2),2)

        self.neighbourhood_func = tf.exp(tf.divide(tf.negative(tf.cast(
            self.bmu_distance_squares, "float32")), tf.multiply(
            tf.square(tf.multiply(radius, self.gaussian_std)), 2)))

        self.learning_rate_op = tf.multiply(self.neighbourhood_func, alpha)

        self.numerator = tf.reduce_sum(
            tf.multiply(tf.expand_dims(self.learning_rate_op, axis=-1),
                        tf.expand_dims(self.input, axis=1)), axis=0)

        self.denominator = tf.expand_dims(
            tf.reduce_sum(self.learning_rate_op, axis=0) + float(1e-20), axis=-1)

        self.new_weights = tf.div(self.numerator, self.denominator)
        self.update = [tf.assign(self.map, self.new_weights)]

        return self.update, tf.reduce_mean(self.grad_pass, 1)


# For plotting the images
from matplotlib import pyplot as plt

# Training inputs for RGBcolors
colors = np.array(
    [[0., 0., 0.],
     [0., 0., 1.],
     [0., 0., 0.5],
     [0.125, 0.529, 1.0],
     [0.33, 0.4, 0.67],
     [0.6, 0.5, 1.0],
     [0., 1., 0.],
     [1., 0., 0.],
     [0., 1., 1.],
     [1., 0., 1.],
     [1., 1., 0.],
     [1., 1., 1.],
     [.33, .33, .33],
     [.5, .5, .5],
     [.66, .66, .66]])
color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# Train a 20x30 SOM with 400 iterations
som = SOM_Layer(m=20,n=30,dim=3,num_epoch=400,learning_rate_som=0.001,radius_factor=4,gaussian_std=1.0)
som.train(colors)

# Get output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped = som.map_vects(colors)

# Plot
plt.imshow(image_grid)
plt.title('Color SOM')
for i, m in enumerate(mapped):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
