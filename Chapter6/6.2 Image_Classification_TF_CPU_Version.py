
# Logging the results
import logging
LOG_FILENAME = 'Image_Classification_CPU.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)

# CPU Version - CIFAR-10 dataset
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Method 2 : Use Keras inbuilt functions
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train_2 = np.zeros((np.shape(y_train)[0], 10))
y_train_2[np.arange(np.shape(y_train)[0]), np.ndarray.flatten(y_train)] = 1

y_test_2 = np.zeros((np.shape(y_test)[0], 10))
y_test_2[np.arange(np.shape(y_test)[0]), np.ndarray.flatten(y_test)] = 1

num_dpoints = 20000

x_train = x_train[:num_dpoints, :]
y_train_2 = y_train_2[:num_dpoints, :]

print(x_train.shape,y_train_2.shape,x_test.shape,y_test_2.shape)

# category to class name dictionary
category_dictionary = {0:'airplane',1:'automobile',2:'bird',3:'cat',4:'deer',
                       5:'dog',6:'frog',7:'horse',8:'ship',9:'truck' }

# plotting first 12 images
images_to_show = 12
plt.figure(figsize=(10, 5))

for xx in range(images_to_show):
    # Number of rows = 3, columns = 4
    plt.subplot(3,4,xx+1)
    plt.imshow(x_train[xx])
    plt.title(category_dictionary[int(y_train[xx])])
plt.show()




def CNN_Classification_CPU_TF(_x_data,_keep_prob):

    conv1_filter = tf.Variable(tf.random_uniform([5,5,3,64]),name="conv1_filter")
    conv2_filter = tf.Variable(tf.random_uniform([5,5,64,128]),name="conv2_filter")
    conv3_filter = tf.Variable(tf.random_uniform([3,3,128,256]),name="conv3_filter")

    # First layer with convolution and pooling
    conv1 = tf.nn.conv2d(_x_data,conv1_filter,strides=[1,1,1,1],padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # Second layer with convolution and pooling
    conv2 = tf.nn.conv2d(conv1_bn,conv2_filter,strides=[1,1,1,1],padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # Third layer with convolution and pooling
    conv3 = tf.nn.conv2d(conv2_bn,conv3_filter,strides=[1,1,1,1],padding='SAME')
    conv3= tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # Flattening layer
    flat_1 = tf.contrib.layers.flatten(conv3_bn)

    # Fully connected layer 1
    full_1 = tf.contrib.layers.fully_connected(inputs=flat_1,num_outputs=256,activation_fn=tf.nn.relu)
    full_1 = tf.nn.dropout(full_1,keep_prob=_keep_prob)
    full_1 = tf.layers.batch_normalization(full_1)

    # Fully connected layer 2
    full_2 =  tf.contrib.layers.fully_connected(inputs=full_1,num_outputs=128,activation_fn=tf.nn.relu)
    full_2 = tf.nn.dropout(full_2,keep_prob=_keep_prob)
    full_2 = tf.layers.batch_normalization(full_2)

    # Final layer with 10 classes of categories
    out = tf.contrib.layers.fully_connected(inputs=full_2,num_outputs=10,activation_fn=None)
    return out


# Parameters
epochs = 10
batch_size = 512
keep_probability = 0.7
learning_rate = 0.001
num_classes=10

# placeholder for variables
xs = tf.placeholder(tf.float32, [None, 32, 32, 3], name="Input_data")
ys = tf.placeholder(tf.float32, [None, num_classes], name="Output_data")
outputs = CNN_Classification_CPU_TF(xs, keep_probability)
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=ys))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

# placeholder for Accuracies
correct_pred = tf.equal(tf.argmax(outputs, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
   # Writing model to tensorboard
    writer = tf.summary.FileWriter("/tmp/tf_imageclasslogs",sess.graph)

    for epoch in range(epochs):
        batch_count = int(np.ceil(x_train.shape[0]/batch_size))
        for i in range(batch_count):
            batch_x = x_train[(i * batch_size): ((i + 1) * batch_size), :]
            batch_y = y_train_2[(i * batch_size): ((i + 1) * batch_size), :]
            trc, trs = sess.run([train_op, cost_op], feed_dict={xs: batch_x, ys: batch_y})

            batch_train_acc = sess.run(accuracy, feed_dict={xs: batch_x, ys: batch_y})
            # Logging accuracies into log
            logging.info(" Epoch :"+ str(epoch + 1) + " batch :" + str(i + 1) + " Train accuracy :" +str(batch_train_acc))
            # Printing accuracies in console
            print(" Epoch :", epoch + 1, "batch :", i + 1, " Train accuracy :",batch_train_acc)

        trcost = sess.run(cost_op,feed_dict={xs:x_train, ys:y_train_2})
        tstcost = sess.run(cost_op,feed_dict={xs:x_test, ys:y_test_2})
        # Writing loss values to tensorboard at each epoch
        tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=trcost)])
        writer.add_summary(tr_summary, global_step=epoch)
        tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=tstcost)])
        writer.add_summary(tst_summary, global_step=epoch)

        # Logging and Printing train and test accuracies
        full_batch_trng_acc = sess.run(accuracy, feed_dict={xs: x_train, ys: y_train_2})
        logging.info(" Full batch Training accuracy Epoch :" + str(epoch + 1) +  " Train accuracy :" + str(full_batch_trng_acc))
        print(" Full batch Training accuracy Epoch :" + str(epoch + 1) +  " Train accuracy :" + str(full_batch_trng_acc))

        full_batch_test_acc = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test_2})
        logging.info(" Full batch Test accuracy Epoch :" + str(epoch + 1) +  " Test accuracy :" + str(full_batch_trng_acc))
        print(" Full batch Test accuracy Epoch :" + str(epoch + 1) +  " Test accuracy :" + str(full_batch_trng_acc))

    # Logging and printing final accuracies
    logging.info("Optimization completed!")
    print("Optimization completed!")

    final_trng_acc = sess.run(accuracy, feed_dict={xs: x_train, ys: y_train_2})
    logging.info(" Final  Training accuracy  :" +  str(final_trng_acc))
    print(" Final  Training accuracy  :" +  str(final_trng_acc))

    final_test_acc = sess.run(accuracy, feed_dict={xs: x_test, ys: y_test_2})
    logging.info(" Final Testing accuracy  :" + str(final_test_acc))
    print(" Final Testing accuracy  :" + str(final_test_acc))


