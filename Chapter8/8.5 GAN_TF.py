

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import pandas as pd
from keras.datasets import mnist

# Importing MNIST data
(x_train,y_train),(_,_) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],784)


# Select the number you would like to train upon
# here required digit has been selected
# Note: By training on all the digits will create noise
x_train = x_train[y_train.flatten()==2]


# Initilization function
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')
G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')
G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

# Generator function
def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob



# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')
D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
theta_D = [D_W1, D_W2, D_b1, D_b2]

# Discriminator function
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# Plotting function for saving samples from random noise
def plot(_samples):
    _fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for _i, sample in enumerate(_samples):
        ax = plt.subplot(gs[_i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return _fig


# Creating generative sample from noise
G_sample = generator(Z)
# Discriminative function on orginal and generated data
D_real, D_logit_real = discriminator(X)
D_gen, D_logit_gen = discriminator(G_sample)

# Combining both discriminative losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.zeros_like(D_logit_gen)))
D_loss = D_loss_real + D_loss_gen

# Generative loss
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_gen, labels=tf.ones_like(D_logit_gen)))

# Optimization on minimization of losses
lr = 0.01
D_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=theta_G)


# Model parameters
batch_size = 128
Z_dim = 100
training_epochs = 100

# Session initiation
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('output/images/'):
    os.makedirs('output/images/')


# Creating dummy array for saving values
dummyarray = np.empty((training_epochs//10,3))
results_loss_df = pd.DataFrame(dummyarray)
results_loss_df.columns=['Iter','Gen_loss','Disc_loss']

i = 0

for it in range(training_epochs):
    # saving intermediate generated samples and plotting loss values
    if it % 10 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('output/images/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        plt.close(fig)
        # Computing on whole data
        _, D_loss_full = sess.run([D_solver, D_loss], feed_dict={X:x_train, Z: sample_Z(x_train.shape[0], Z_dim)})
        _, G_loss_full = sess.run([G_solver, G_loss], feed_dict={Z:sample_Z(x_train.shape[0], Z_dim)})
        print('Iter: {}'.format(it))
        print('Generator_full_loss: {:.4}'.format(G_loss_full))
        print('Discriminator_full_loss: {:.4}'.format(D_loss_full))
        results_loss_df.loc[i,'Iter'] = it
        results_loss_df.loc[i,'Gen_loss'] = G_loss_full
        results_loss_df.loc[i,'Disc_loss'] = D_loss_full
        i+=1

    # Training both generator and discriminator
    num_batches = x_train.shape[0]//batch_size
    for nb in range(num_batches):
        batch_images = x_train[(nb*batch_size):((nb+1)*batch_size),:]
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: batch_images, Z: sample_Z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})
        print("epoch: ",it,"batch: ",nb,"gen loss: ",G_loss_curr,"disc loss: ",D_loss_curr)


# Plotting losses over epochs
plt.figure()
plt.xlabel('Iterations')
plt.ylabel('Losses')
plt.plot(results_loss_df['Iter'],results_loss_df['Gen_loss'])
plt.plot(results_loss_df['Iter'],results_loss_df['Disc_loss'])
plt.legend(loc='upper right')
plt.savefig('output/images/losses.png')
plt.show()
plt.close()


print("Completed")

