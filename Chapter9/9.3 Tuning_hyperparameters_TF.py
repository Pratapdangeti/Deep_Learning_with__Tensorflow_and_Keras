


# Multi Classification model using TensorFlow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits


# Loading data
digits = load_digits()
X = digits.data
y = digits.target


# data preprocessing
input_dim = 64
# layers
layer_1_neurons = 10
layer_2_neurons = 10
num_classes = 10

x_vars_stdscle = StandardScaler().fit_transform(X)
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y_,train_size = 0.7,random_state=42)



def DNN_Multi_Classification_TF(_x):
    # Layer 1
    W1=tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]))
    b1= tf.Variable(tf.zeros([layer_1_neurons]))
    layer_1 = tf.add(tf.matmul(_x,W1),b1)
    layer_1 = tf.nn.relu(layer_1)

    # Layer 2
    W2=tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]))
    b2=tf.Variable(tf.zeros([layer_2_neurons]))
    layer_2 = tf.add(tf.matmul(layer_1,W2),b2)
    layer_2 = tf.nn.relu(layer_2)

    # Output Layer
    WO=tf.Variable(tf.random_uniform([layer_2_neurons,num_classes]))
    bo=tf.Variable(tf.zeros([num_classes]))
    output_layer = tf.add(tf.matmul(layer_2,WO),bo)
    return output_layer


# Code starts here
xs = tf.placeholder(tf.float32,[None,input_dim],name="Input_data")
ys = tf.placeholder(tf.float32,[None,num_classes],name="output_data")

# Construct model
output = DNN_Multi_Classification_TF(xs)


# Define loss and output
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=ys))


# Tuning 1 -  Batch size

training_epochs = 60
learning_rate = 0.01
batch_size_list = [15,32,64,128,256,512]
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

for batch_size in batch_size_list:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            batch_count = int(x_train.shape[0]/batch_size)
            for i in range(batch_count):
                batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
                batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
                trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})
        act_amax = tf.argmax(ys,1)
        pred_amax = tf.argmax(output, 1)
        print("Batch size: ",batch_size,", Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4),", Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))
        sess.close()



# Tuning 2 - Training epoch vs. Learning rate
"""
batch_size = 64

# Tuning batch size
training_epochs_list = [30,50,100,200,300]
learning_rate_list = [0.1,0.04,0.02,0.01]

best_test_accuracy = 0.0

for learning_rate in learning_rate_list:
    for training_epochs in training_epochs_list:
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(training_epochs):
                batch_count = int(x_train.shape[0]/batch_size)
                for i in range(batch_count):
                    batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
                    batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
                    trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})
            act_amax = tf.argmax(ys,1)
            pred_amax = tf.argmax(output, 1)

            train_acc = round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4)
            test_acc = round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4)

            if test_acc > best_test_accuracy:
                best_test_accuracy = test_acc
                print("Training epochs: ",training_epochs,", Learning rate",learning_rate,", Train Accuracy : ",train_acc,", Test Accuracy : ",test_acc)
            sess.close()

"""


# Tuning 3 - Optimization methods
"""
batch_size = 64
training_epochs = 50


train_op_list = [
    tf.train.GradientDescentOptimizer(learning_rate=0.04,name="Gradient_descent_optimizer").minimize(cost_op),
    tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9,name="Momentum_optimizer").minimize(cost_op),
    tf.train.AdagradDAOptimizer(learning_rate=0.01,global_step=tf.constant(1,dtype=tf.int64),name="Adagrad_optimizer").minimize(cost_op),
    tf.train.AdadeltaOptimizer(learning_rate=0.01,rho=0.95,epsilon=1e-08,name="Adadelta_optimizer").minimize(cost_op),
    tf.train.RMSPropOptimizer(learning_rate=0.04,decay=0.9,epsilon=1e-10,momentum=0,name="RMSProp_optimizer").minimize(cost_op),
    tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08,name="Adam_optimizer").minimize(cost_op)
]


for train_op in train_op_list:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            batch_count = int(x_train.shape[0]/batch_size)
            for i in range(batch_count):
                batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
                batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
                trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})
        act_amax = tf.argmax(ys,1)
        pred_amax = tf.argmax(output, 1)
        print(train_op.name,", Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4),", Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))
        sess.close()
"""






# Tuning 4 - dropout rate
"""
def DNN_Multi_Classification_WDOR_TF(_x,_keep_prob):
    # Layer 1
    W1=tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]))
    b1= tf.Variable(tf.zeros([layer_1_neurons]))
    layer_1 = tf.add(tf.matmul(_x,W1),b1)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1,keep_prob=_keep_prob)

    # Layer 2
    W2=tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]))
    b2=tf.Variable(tf.zeros([layer_2_neurons]))
    layer_2 = tf.add(tf.matmul(layer_1,W2),b2)
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2,keep_prob=_keep_prob)

    # Output Layer
    WO=tf.Variable(tf.random_uniform([layer_2_neurons,num_classes]))
    bo=tf.Variable(tf.zeros([num_classes]))
    output_layer = tf.add(tf.matmul(layer_2,WO),bo)
    return output_layer


batch_size = 64
training_epochs = 50
learning_rate = 0.04

dropout_rate_list = [0.1,0.3,0.5,0.8]

for _drop_out in dropout_rate_list:

    # Construct model
    output_wdo = DNN_Multi_Classification_WDOR_TF(xs,_keep_prob=1-_drop_out)
    # Define loss and output
    cost_op_wdo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_wdo, labels=ys))
    train_op_wdo = tf.train.RMSPropOptimizer(learning_rate=0.04, decay=0.9, epsilon=1e-10, momentum=0,
                                         name="RMSProp_optimizer").minimize(cost_op_wdo)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            batch_count = int(x_train.shape[0]/batch_size)
            for i in range(batch_count):
                batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
                batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
                trc,trs = sess.run([train_op_wdo,cost_op_wdo],feed_dict={xs:batch_x, ys:batch_y})
        act_amax = tf.argmax(ys,1)
        pred_amax = tf.argmax(output_wdo, 1)
        print("Dropout rate: ",_drop_out,", Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4),", Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))
        sess.close()
"""



# Tuning 5 - Activation functions
"""
def DNN_Multi_Classification_WAF_TF(_x,_act_func):
    # Layer 1
    W1=tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]))
    b1= tf.Variable(tf.zeros([layer_1_neurons]))
    layer_1 = tf.add(tf.matmul(_x,W1),b1)
    layer_1 = _act_func(layer_1)
    layer_1 = tf.nn.dropout(layer_1,keep_prob=0.9)
    # Layer 2
    W2=tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]))
    b2=tf.Variable(tf.zeros([layer_2_neurons]))
    layer_2 = tf.add(tf.matmul(layer_1,W2),b2)
    layer_2 = _act_func(layer_2)
    layer_2 = tf.nn.dropout(layer_2,keep_prob=0.9)
    # Output Layer
    WO=tf.Variable(tf.random_uniform([layer_2_neurons,num_classes]))
    bo=tf.Variable(tf.zeros([num_classes]))
    output_layer = tf.add(tf.matmul(layer_2,WO),bo)
    return output_layer


batch_size = 64
training_epochs = 50
learning_rate = 0.04

act_func_list = [tf.nn.relu,tf.nn.sigmoid,tf.nn.tanh]

for act_func in act_func_list:

    # Construct model
    output_waf = DNN_Multi_Classification_WAF_TF(xs,_act_func=act_func)
    # Define loss and output
    cost_op_waf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_waf, labels=ys))
    train_op_waf = tf.train.RMSPropOptimizer(learning_rate=0.04, decay=0.9, epsilon=1e-10, momentum=0,
                                         name="RMSProp_optimizer").minimize(cost_op_waf)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            batch_count = int(x_train.shape[0]/batch_size)
            for i in range(batch_count):
                batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
                batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
                trc,trs = sess.run([train_op_waf,cost_op_waf],feed_dict={xs:batch_x, ys:batch_y})
        act_amax = tf.argmax(ys,1)
        pred_amax = tf.argmax(output_waf,1)
        print("Activation function: ",act_func.__name__,", Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4),", Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))
        sess.close()

"""

# Tuning 6 - Number of layers and neurons in each layer
"""
def DNN_Multi_Class_Nlyrsnrn_TF(_x,network_config, name="neuralnet"):
    layers = {}
    layers_compute = {}
    with tf.name_scope(name):
        for _i in range(1, len(network_config)):
            new_layer = {'weights': tf.Variable(tf.random_normal([network_config[_i - 1], network_config[_i]], 0, 0.1)),
                        'biases': tf.Variable(tf.random_normal([network_config[_i]], 0, 0.1))}
            layers[_i-1] = new_layer
            with tf.name_scope("weights"):
                tf.summary.histogram("w_l"+str(_i)+"_summary",new_layer['weights'])
            with tf.name_scope("biases"):
                tf.summary.histogram("b_l"+str(_i)+"_summary",new_layer['biases'])
            lngth = tf.add(tf.matmul(_x if _i == 1 else layers_compute[_i - 2], layers[_i - 1]['weights']), layers[_i - 1]['biases'])
            with tf.name_scope(name):
                lngth = tf.nn.relu(lngth) if _i != len(network_config) - 1 else lngth
            layers_compute[_i-1] = lngth
    final_layer = len(layers_compute)-1
    return layers_compute[final_layer]


batch_size = 64
training_epochs = 50
learning_rate = 0.04

architecture_list = [[input_dim, 10, 10, num_classes],
                     [input_dim,10, 10, 10, num_classes],
                     [input_dim, 30, 20, num_classes]]

for architecture in architecture_list:
    output_Nlyrsnrn = DNN_Multi_Class_Nlyrsnrn_TF(xs,network_config=architecture)
    cost_Nlyrsnrn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_Nlyrsnrn,labels=ys))
    train_op_Nlyrsnrn = tf.train.RMSPropOptimizer(learning_rate=0.04, decay=0.9, epsilon=1e-10, momentum=0,
                                             name="RMSProp_optimizer").minimize(cost_Nlyrsnrn)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epochs):
            batch_count = int(x_train.shape[0] / batch_size)
            for i in range(batch_count):
                batch_x = x_train[(i * batch_size): ((i + 1) * batch_size), :]
                batch_y = y_train[(i * batch_size): ((i + 1) * batch_size), :]
                trc, trs = sess.run([train_op_Nlyrsnrn, cost_Nlyrsnrn], feed_dict={xs: batch_x, ys: batch_y})
        act_amax = tf.argmax(ys, 1)
        pred_amax = tf.argmax(output_Nlyrsnrn,1)
        print("Architecture: ", architecture, ", Train Accuracy : ",
              round(accuracy_score(act_amax.eval({ys: y_train}), pred_amax.eval({xs: x_train})), 4), ", Test Accuracy : ",
              round(accuracy_score(act_amax.eval({ys: y_test}), pred_amax.eval({xs: x_test})), 4))
        sess.close()
"""



print("Completed!")

