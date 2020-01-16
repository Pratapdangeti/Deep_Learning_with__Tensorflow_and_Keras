


import numpy as np
import tensorflow as tf


# Use Keras inbuilt functions for importing data
from keras.datasets import cifar10



def CNN_Classification_GPU_TF(_x_data,_keep_prob):
    # Reusing variables for later batch updates
    # Reusing variables does work with variable scope only
    with tf.variable_scope("conv",reuse=tf.AUTO_REUSE):
        conv1_filter = tf.Variable(tf.random_uniform([5,5,3,64]),name="conv1_filter")
        conv2_filter = tf.Variable(tf.random_uniform([5,5,64,128]),name="conv2_filter")
        conv3_filter = tf.Variable(tf.random_uniform([3,3,128,256]),name="conv3_filter")

        conv1 = tf.nn.conv2d(_x_data,conv1_filter,strides=[1,1,1,1],padding='SAME')
        conv1 = tf.nn.relu(conv1)
        conv1_pool = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        conv1_bn = tf.layers.batch_normalization(conv1_pool)

        conv2 = tf.nn.conv2d(conv1_bn,conv2_filter,strides=[1,1,1,1],padding='SAME')
        conv2 = tf.nn.relu(conv2)
        conv2_pool = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        conv2_bn = tf.layers.batch_normalization(conv2_pool)

        conv3 = tf.nn.conv2d(conv2_bn,conv3_filter,strides=[1,1,1,1],padding='SAME')
        conv3= tf.nn.relu(conv3)
        conv3_pool = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        conv3_bn = tf.layers.batch_normalization(conv3_pool)

        flat_1 = tf.contrib.layers.flatten(conv3_bn)

        full_1 = tf.contrib.layers.fully_connected(inputs=flat_1,num_outputs=256,activation_fn=tf.nn.relu)
        full_1 = tf.nn.dropout(full_1,keep_prob=_keep_prob)
        full_1 = tf.layers.batch_normalization(full_1)

        full_2 =  tf.contrib.layers.fully_connected(inputs=full_1,num_outputs=128,activation_fn=tf.nn.relu)
        full_2 = tf.nn.dropout(full_2,keep_prob=_keep_prob)
        full_2 = tf.layers.batch_normalization(full_2)

        out = tf.contrib.layers.fully_connected(inputs=full_2,num_outputs=10,activation_fn=None)
    return out


# Parameters to tune
epochs = 10
batch_size = 512
keep_probability = 0.7
learning_rate = 0.001
num_classes=10
# number of data points retained for training from original 50k observations
num_dpoints = 20000


# Heterogeneous computing with both cpu & gpu devices
# Initially all the data stored in CPU later batches moved to GPU
# Finally testing on trained model does happen on CPU
with tf.device('/cpu:0'):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train_2 = np.zeros((np.shape(y_train)[0],10))
    y_train_2[np.arange(np.shape(y_train)[0]),np.ndarray.flatten(y_train)]=1
    y_test_2 = np.zeros((np.shape(y_test)[0],10))
    y_test_2[np.arange(np.shape(y_test)[0]),np.ndarray.flatten(y_test)]=1

    x_train = x_train[:num_dpoints,:]
    x_test = x_test[:num_dpoints,:]

    y_train_2 = y_train_2[:num_dpoints,:]
    y_test_2 = y_test_2[:num_dpoints,:]

    # Following variables are for batch processing
    xs = tf.placeholder(tf.float32,[None,32,32,3],name="Input_data")
    ys = tf.placeholder(tf.float32,[None,10],name="Output_data")

    # Following variables are for testing trained model on entire dataset at the end of each epoch
    axs = tf.placeholder(tf.float32,[None,32,32,3],name="Input_data_1")
    ays = tf.placeholder(tf.float32,[None,10],name="Output_data_1")
    all_outputs = CNN_Classification_GPU_TF(axs, keep_probability)
    all_correct_pred = tf.equal(tf.argmax(all_outputs, 1), tf.argmax(ays, 1))
    all_accuracy = tf.reduce_mean(tf.cast(all_correct_pred, tf.float32))

    # Following code activates GPU device for batch processing
    with tf.device('/gpu:0'):
        _x = xs
        _y = ys
        # To activate all devices of heterogeneous computing
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            outputs = CNN_Classification_GPU_TF(_x, keep_probability)
            cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=_y))
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, name="adam2").minimize(cost_op)

            correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            sess.run(tf.global_variables_initializer())

            for epc in range(epochs):
                _batch_count = int(np.ceil(x_train.shape[0] / batch_size))
                for j in range(_batch_count):
                    batch_x = x_train[j*batch_size: (j+1)*batch_size]
                    batch_y = y_train_2[j*batch_size:(j+1)*batch_size]
                    # Following is the main training function to optimize cost on batches
                    sess.run(train_op,feed_dict={_x:batch_x,_y:batch_y})
                    # to print train accuracy for each batch
                    print("Train batch accuracy :",sess.run(accuracy,feed_dict={_x:batch_x,_y:batch_y}))

                # At the end of each epoch, train and test error will be printed on entire dataset
                with tf.device('/cpu:0'):
                    print("epoch :",epc+1,"Trn acc :",sess.run(all_accuracy,feed_dict={axs:x_train,ays:y_train_2}),"Test acc :", sess.run(all_accuracy, feed_dict={axs: x_test, ays: y_test_2}))


print("Optimization finished")



