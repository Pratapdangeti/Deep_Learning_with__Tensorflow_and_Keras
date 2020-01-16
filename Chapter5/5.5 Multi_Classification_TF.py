
# Multi Classification model using TensorFlow

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


# Loading data
digits = load_digits()
X = digits.data
y = digits.target

# printing the first data point
print ("\nPrinting first digit")
plt.matshow(digits.images[0])
plt.show()

# data preprocessing
input_dim = 64
num_classes = 10
x_vars_stdscle = StandardScaler().fit_transform(X)
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y_,train_size = 0.7,random_state=42)


# layers
layer_1_neurons = 10
layer_2_neurons = 10
learning_rate = 0.01

weights = {
    'W1':tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]),name="W1"),
    'W2':tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]),name="W2"),
    'WO':tf.Variable(tf.random_uniform([layer_2_neurons,num_classes]),name="WO")
}

biases = {
    'b1':tf.Variable(tf.zeros([layer_1_neurons]),name="b1"),
    'b2':tf.Variable(tf.zeros([layer_2_neurons]),name="b2"),
    'bo':tf.Variable(tf.zeros([num_classes]),name="bo")
}


def DNN_Multi_Classification_TF(_x, _weights, _biases):
    # Layer 1
    layer_1 = tf.add(tf.matmul(_x,_weights['W1']),_biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Layer 2
    layer_2 = tf.add(tf.matmul(layer_1,_weights['W2']),_biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output Layer
    output_layer = tf.add(tf.matmul(layer_2,_weights['WO']),_biases['bo'])
    return output_layer


# Code starts here
xs = tf.placeholder(tf.float32,[None,input_dim],name="Input_data")
ys = tf.placeholder(tf.float32,[None,num_classes],name="output_data")

# Construct model
output = DNN_Multi_Classification_TF(xs,weights,biases)


# Define loss and output
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=ys))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)


training_epochs = 60
batch_size = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/tmp/tf_multiclasslogs",sess.graph)

    for epoch in range(training_epochs):

        batch_count = int(x_train.shape[0]/batch_size)
        for i in range(batch_count):
            batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
            batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]

            trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})

            print("Epoch :",epoch,"batch :",i," Train Cost :",sess.run(cost_op,feed_dict={xs:x_train, ys:y_train}),
                  "Test Cost :", sess.run(cost_op, feed_dict={xs: x_test, ys: y_test}))

        trcost = sess.run(cost_op,feed_dict={xs:x_train, ys:y_train})
        tstcost = sess.run(cost_op,feed_dict={xs:x_test, ys:y_test})
        # Writing loss values to tensorboard at each epoch
        tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=trcost)])
        # +1 added to epoch only to represent the step
        writer.add_summary(tr_summary, global_step=epoch)
        tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=tstcost)])
        writer.add_summary(tst_summary, global_step=epoch)
    writer.close()
    print("Optimization Finished!")

    act_amax = tf.argmax(ys,1)
    pred_amax = tf.argmax(output, 1)


    print("Multi Classification Train Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train}) ))
    print("Multi Classification Test Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test}) ))

    print("Multi Classification Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4))
    print("Multi Classification Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))

    sess.close()
























