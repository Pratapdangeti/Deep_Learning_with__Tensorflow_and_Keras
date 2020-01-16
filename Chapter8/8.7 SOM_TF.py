
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt



#2-D Self-Organizing Map with function of Gaussian Neighbourhood
class SOM:
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        m X n: Dimensions of the SOM.
        'dim': Dimensionality of the training inputs.
        'n_iterations': Number of iterations
        'alpha':learning rate. Default value is 0.3
        'sigma': Initial neighbourhood value(or radius of influence) of the
         Best Matching Unit (BMU) while training. Default value taken as half of max(m, n)
        """
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        #Initialize graph
        self._graph = tf.Graph()
        with self._graph.as_default():

            # Randomly initialized weightage vectors for all neurons [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal([m*n,dim]))

            # Matrix of size [m*n, 2] for SOM grid locations of neurons
            self._location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))

            #Placeholders for training inputs
            self._vect_input = tf.placeholder(tf.float32,[dim])

            # Iteration number
            self._iter_input = tf.placeholder(tf.float32)

            # Computing the Best Matching Unit(BMU) & index, given a vector based on Euclidean distance
            # between neuron and input
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),0)

            # Extract the location of the Best matching unit (BMU) based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),np.array([[0, 1]]))
            slice_input = tf.cast(slice_input,tf.int32)
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,tf.constant(np.array([1, 2]))),[2])

            # Compute the updated alpha and sigma values based on iteration
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input,self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects,
                                        tf.stack([bmu_loc for i in range(m*n)])),2),1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares,tf.float32),
                                                           tf.pow(_sigma_op,2))))
            learning_rate_op = tf.multiply(_alpha_op,neighbourhood_func)

            # Update the weightage vectors of all neurons based on a particular input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)])
            weightage_delta = tf.multiply(learning_rate_multiplier,
                         tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,new_weightages_op)

            #Initiate session
            self._sess = tf.Session()
            #Initialize variables
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        # generates 2-D locations in the map
        for _i in range(m):
            for _j in range(n):
                yield np.array([_i, _j])

    def train(self, input_vects):
        #Training of SOM
        for _iter in range(self._n_iterations):
            # Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,self._iter_input: _iter})
            print("Iteration :",_iter)
        # Computes the centroids of grid
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for _i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[_i])
        self._centroid_grid = centroid_grid

    def get_centroids(self):
        # Returns a list of 'm' lists, with each inner list the 'n' corresponding
        # centroid locations as 1-D NumPy arrays.
        return self._centroid_grid

    def map_vects(self, input_vects):
        #Maps every input vector to the relevant neuron in the SOM grid.
        maps_return_vecs = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-self._weightages[x]))
            maps_return_vecs.append(self._locations[min_index])
        return maps_return_vecs



# Training inputs for RGBcolors of arrays
# of each channel of RGB
colors_index = np.array(
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


color_names = ['black', 'blue', 'darkblue', 'skyblue','greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white','darkgrey', 'mediumgrey', 'lightgrey']


# Train a 20x30 SOM with 40 iterations
som = SOM(20, 30, 3, 40)
som.train(colors_index)

# Get output grid
image_grid = som.get_centroids()

# Map colours to their closest neurons
mapped_index = som.map_vects(colors_index)


# Plotting the 2D
plt.imshow(image_grid)
#plt.title('SOM of Colors')
for i, m in enumerate(mapped_index):
    plt.text(m[1], m[0], color_names[i], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))



print("Completed!")

