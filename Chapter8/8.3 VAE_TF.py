

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Make sure to install tensorflow version of 1.13 or later to
# make probability distributions work
import tensorflow_probability as tfp
tfd = tfp.distributions

# Make Encoder and posterior
# will try to be normal distrn with 0 and 1
def create_encoder(_data, code_size):
    x = tf.layers.flatten(_data)
    x = tf.layers.dense(x,250,tf.nn.relu)
    x = tf.layers.dense(x,250, tf.nn.relu)
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)

# Creating prior of normal distribution with
# 0 mean and 1 variance
def create_prior(code_size):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)

# Decoder to reconstruct Bernoulli distribution
def create_decoder(_code, data_shape):
    x = _code
    x = tf.layers.dense(x,250,tf.nn.relu)
    x = tf.layers.dense(x,250,tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)
    return tfd.Independent(tfd.Bernoulli(logit),2)

#  plotting latent vectors
def plot_codes(ax, codes, labels):
    ax.scatter(codes[:, 0], codes[:, 1], s=2, c=labels, alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min() - .1, codes.max() + .1)
    ax.set_ylim(codes.min() - .1, codes.max() + .1)
    ax.tick_params(axis='both', which='both', left='off', bottom='off',
      labelleft='off', labelbottom='off')

# plotting output from decoders
def plot_samples(ax,samples):
    for index,sample in enumerate(samples):
        ax[index].imshow(sample, cmap='gray')
        ax[index].axis('off')

data = tf.placeholder(tf.float32,[None,28,28])
# Make_template is a helper function for wrapping any arbitrary function
# so that it does variable sharing
make_encoder = tf.make_template('encoder',create_encoder)
make_decoder = tf.make_template('decoder',create_decoder)

# Define each section of the model
prior = create_prior(code_size=2)
posterior = make_encoder(data, code_size=2)
code = posterior.sample()

# Defining the loss functions
likelihood = make_decoder(code,[28, 28]).log_prob(data)
kl_divergence = tfd.kl_divergence(posterior, prior)

# Objectie to maximize likelihood and minimize divergence loss
# elbo - evidence lower bound
elbo = tf.reduce_mean(likelihood - kl_divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)
samples = make_decoder(prior.sample(10),[28, 28]).mean()


mnist = input_data.read_data_sets('MNIST_data/')
fig, ax = plt.subplots(nrows=5,ncols=11,figsize=(10,10))
with tf.train.MonitoredSession() as sess:
    cntr = 0
    for epoch in range(21):
        feed = {data: mnist.test.images.reshape([-1, 28, 28])}
        test_elbo,test_codes,test_samples = sess.run([elbo,code,samples],feed)
        print('Epoch', epoch, 'elbo', test_elbo)
        # Plotting intermediate results
        if epoch%5==0:
            ax[cntr,0].set_ylabel('Epoch {}'.format(epoch))
            plot_codes(ax[cntr,0],test_codes,mnist.test.labels)
            plot_samples(ax[cntr, 1:],test_samples)
            cntr+=1
        # Training over dataset
        for _ in range(600):
          feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
          sess.run(optimize, feed)

plt.show()
print("Completed!")


