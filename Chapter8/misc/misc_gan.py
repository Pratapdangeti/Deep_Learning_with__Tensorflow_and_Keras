


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0],784)


# Generator model
def generator(_z,reuse=None):
    with tf.variable_scope('gen_env',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=_z,units=1024,activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1,units=1024,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(inputs=hidden2,units=784,activation=tf.nn.tanh)
        return output

# Discriminator model
def discriminator(_x,reuse=None):
    with tf.variable_scope('dis_env',reuse=reuse):
        hidden1 = tf.layers.dense(inputs=_x,units=256,activation=tf.nn.leaky_relu)
        hidden2 = tf.layers.dense(inputs=hidden1,units=256,activation=tf.nn.leaky_relu)
        logits = tf.layers.dense(hidden2,units=1)
        output = tf.sigmoid(logits)
        return output,logits

tf.reset_default_graph()

# Data definition
# real image shape
real_images = tf.placeholder(tf.float32,shape=[None,784])
# noise shape
z = tf.placeholder(tf.float32,shape=[None,100])

# passing noise through generator
G = generator(z)

D_out_real,D_logits_real = discriminator(real_images,reuse=None)
D_out_gen, D_logits_gen = discriminator(G,reuse=True)


def loss_func(_logits,_labels):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=_logits,labels=_labels))

# Smoothing for generalization for 1's values
D_real_loss = loss_func(D_logits_real,tf.ones_like(D_logits_real)*0.9)

# Generated loss
D_gen_loss = loss_func(D_logits_gen,tf.zeros_like(D_logits_gen))
D_loss = D_real_loss+D_gen_loss

G_loss = loss_func(D_logits_gen,tf.ones_like(D_logits_gen))


lr = 0.001

#Do this when multiple networks interact with each other
#returns all variables created(the two variable scopes) and makes trainable true

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'dis_env' in var.name]
g_vars = [var for var in tvars if 'gen_env' in var.name]


D_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(D_loss,var_list=d_vars)
G_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=g_vars)


batch_size = 1000
training_epochs = 10
init=tf.global_variables_initializer()
# generator samples
samples = []

dummyarray = np.empty((training_epochs//10,3))
results_loss_df = pd.DataFrame(dummyarray)
results_loss_df.columns=['Iter','Gen_loss','Disc_loss']


with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("/tmp/tf_ganlogs",sess.graph)

    for epoch in range(training_epochs):
        num_batches = x_train.shape[0]//batch_size
        #num_batches = mnist.train.num_examples//batch_size
        for i in range(num_batches):

            batch_images = x_train[(i*batch_size):((i+1)*batch_size),:]
            batch_images = batch_images * 2 -1

            #batch = mnist.train.next_batch(batch_size)
            #batch_images = batch[0].reshape((batch_size,784))
            #batch_images = batch_images * 2 -1

            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            _,batch_d_loss = sess.run([D_train,D_loss],feed_dict={real_images:batch_images,z:batch_z})
            _,batch_g_loss = sess.run([G_train,G_loss],feed_dict={z:batch_z})
            #print("epoch: ",epoch,"batch: ",i,"Disc loss: ",batch_d_loss,"Gen loss: ",batch_g_loss)

        full_data = x_train
        full_z = np.random.uniform(-1,1,size=(x_train.shape[0],100))
        _, full_d_loss = sess.run([D_train, D_loss], feed_dict={real_images: full_data, z: full_z})
        _, full_g_loss = sess.run([G_train, G_loss], feed_dict={z: full_z})
        print("epoch: ", epoch, "Disc loss: ", full_d_loss, "Gen loss: ", full_g_loss)

        sample_z = np.random.uniform(-1,1,size=(1,100))
        gen_sample = sess.run(generator(z,reuse=True),feed_dict={z:sample_z})
        samples.append(gen_sample)


plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.imshow(samples[0].reshape(28, 28))

plt.subplot(2,2,2)
plt.imshow(samples[int(training_epochs*0.25)].reshape(28, 28))

plt.subplot(2,2,3)
plt.imshow(samples[int(training_epochs*0.5)].reshape(28, 28))

plt.subplot(2,2,4)
plt.imshow(samples[training_epochs-1].reshape(28, 28))


plt.close()




print("Completed!")




