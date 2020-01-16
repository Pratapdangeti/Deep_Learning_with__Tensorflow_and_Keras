

# Deep Auto Encoders
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import tensorflow as tf
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

x_vars_stdscle = StandardScaler().fit_transform(X)
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y,train_size = 0.7,random_state=42)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)



# layers
input_dim = 64
layer_1_neurons = 32
layer_2_neurons = 16
layer_3_neurons = 3
layer_4_neurons = 16
layer_5_neurons = 32
output_dim = 64



weights = {
    'W1':tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]),name="W1"),
    'W2':tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]),name="W2"),
    'W3': tf.Variable(tf.random_uniform([layer_2_neurons, layer_3_neurons]), name="W3"),
    'W4': tf.Variable(tf.random_uniform([layer_3_neurons, layer_4_neurons]), name="W4"),
    'W5': tf.Variable(tf.random_uniform([layer_4_neurons, layer_5_neurons]), name="W5"),
    'WO':tf.Variable(tf.random_uniform([layer_5_neurons,output_dim]),name="WO")
}

biases = {
    'b1':tf.Variable(tf.zeros([layer_1_neurons]),name="b1"),
    'b2':tf.Variable(tf.zeros([layer_2_neurons]),name="b2"),
    'b3': tf.Variable(tf.zeros([layer_3_neurons]), name="b3"),
    'b4': tf.Variable(tf.zeros([layer_4_neurons]), name="b4"),
    'b5': tf.Variable(tf.zeros([layer_5_neurons]), name="b5"),
    'bo':tf.Variable(tf.zeros([output_dim]),name="bo")
}




# returning both decoder end layer and encoder end's middle layer
# Decoder output used for reducing errors with input
# Encoder end's output used for plotting the compressed 3D latent vectors in chart

def Encoder_Decoder_TF(_x, _weights, _biases):
    # Layer 1
    layer_1 = tf.add(tf.matmul(_x,_weights['W1']),_biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Layer 2
    layer_2 = tf.add(tf.matmul(layer_1,_weights['W2']),_biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Layer 3
    layer_3 = tf.add(tf.matmul(layer_2,_weights['W3']),_biases['b3'])
    layer_3 = tf.nn.relu(layer_3)

    # Layer 4
    layer_4 = tf.add(tf.matmul(layer_3,_weights['W4']),_biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    # Layer 5
    layer_5 = tf.add(tf.matmul(layer_4,_weights['W5']),_biases['b5'])
    layer_5 = tf.nn.relu(layer_5)

    # Output Layer
    output_layer = tf.add(tf.matmul(layer_5,_weights['WO']),_biases['bo'])

    #returning both outut and middle layer
    return output_layer,layer_3

# placeholder for variables
xs = tf.placeholder(tf.float32,[None,64],name="Input_data")
ys = tf.placeholder(tf.float32,[None,64],name="output_data")

# Error will be reduced on output from end layer but
# Middle encoder layer will be used for 3D visualization plots
output,enc_output = Encoder_Decoder_TF(xs,weights,biases)
cost_op = tf.reduce_mean(tf.square(output-ys))
train_op =tf.train.AdamOptimizer(0.01).minimize(cost_op)

# Training parameters
training_epochs = 100
batch_size = 256

reduced_X3D = 0

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/tmp/tf_encdeclogs",sess.graph)

    for epoch in range(training_epochs):
        batch_count = int(x_vars_stdscle.shape[0]/batch_size)
        for i in range(batch_count):
            batch_x = x_vars_stdscle[(i*batch_size): ((i+1)*batch_size),:]
            batch_y = x_vars_stdscle[(i*batch_size): ((i+1)*batch_size),:]
            trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})
            print("Epoch :",epoch,"batch :",i," Train Cost :",sess.run(cost_op,feed_dict={xs:x_train, ys:x_train}),
                  "Test Cost :", sess.run(cost_op, feed_dict={xs:x_test, ys:x_test}))

    writer.close()
    print("Optimization Finished!")

    reduced_X3D = sess.run(enc_output,feed_dict={xs:x_vars_stdscle})
    print("Encoder-Decoder output shape :",reduced_X3D.shape)



# 3-D Plotting functions
zero_x,zero_y,zero_z = [],[],[]
one_x,one_y,one_z = [],[],[]
two_x,two_y,two_z = [],[],[]
three_x,three_y,three_z = [],[],[]
four_x,four_y,four_z = [],[],[]
five_x,five_y,five_z = [],[],[]
six_x,six_y,six_z = [],[],[]
seven_x,seven_y,seven_z = [],[],[]
eight_x,eight_y,eight_z = [],[],[]
nine_x,nine_y,nine_z = [],[],[]

for i in range(len(reduced_X3D)):
    if y[i] == 10:
        continue

    elif y[i] == 0:
        zero_x.append(reduced_X3D[i][0])
        zero_y.append(reduced_X3D[i][1])
        zero_z.append(reduced_X3D[i][2])

    elif y[i] == 1:
        one_x.append(reduced_X3D[i][0])
        one_y.append(reduced_X3D[i][1])
        one_z.append(reduced_X3D[i][2])

    elif y[i] == 2:
        two_x.append(reduced_X3D[i][0])
        two_y.append(reduced_X3D[i][1])
        two_z.append(reduced_X3D[i][2])

    elif y[i] == 3:
        three_x.append(reduced_X3D[i][0])
        three_y.append(reduced_X3D[i][1])
        three_z.append(reduced_X3D[i][2])

    elif y[i] == 4:
        four_x.append(reduced_X3D[i][0])
        four_y.append(reduced_X3D[i][1])
        four_z.append(reduced_X3D[i][2])

    elif y[i] == 5:
        five_x.append(reduced_X3D[i][0])
        five_y.append(reduced_X3D[i][1])
        five_z.append(reduced_X3D[i][2])

    elif y[i] == 6:
        six_x.append(reduced_X3D[i][0])
        six_y.append(reduced_X3D[i][1])
        six_z.append(reduced_X3D[i][2])

    elif y[i] == 7:
        seven_x.append(reduced_X3D[i][0])
        seven_y.append(reduced_X3D[i][1])
        seven_z.append(reduced_X3D[i][2])

    elif y[i] == 8:
        eight_x.append(reduced_X3D[i][0])
        eight_y.append(reduced_X3D[i][1])
        eight_z.append(reduced_X3D[i][2])

    elif y[i] == 9:
        nine_x.append(reduced_X3D[i][0])
        nine_y.append(reduced_X3D[i][1])
        nine_z.append(reduced_X3D[i][2])

# 3- Dimensional plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(zero_x, zero_y, zero_z, c='r', marker='x', label='zero')
ax.scatter(one_x, one_y, one_z, c='g', marker='+', label='one')
ax.scatter(two_x, two_y, two_z, c='b', marker='s', label='two')

ax.scatter(three_x, three_y, three_z, c='m', marker='*', label='three')
ax.scatter(four_x, four_y, four_z, c='c', marker='h', label='four')
ax.scatter(five_x, five_y, five_z, c='r', marker='D', label='five')

ax.scatter(six_x, six_y, six_z, c='y', marker='8', label='six')
ax.scatter(seven_x, seven_y, seven_z, c='k', marker='*', label='seven')
ax.scatter(eight_x, eight_y, eight_z, c='r', marker='x', label='eight')

ax.scatter(nine_x, nine_y, nine_z, c='b', marker='D', label='nine')

ax.set_xlabel('Latent Feature 1', fontsize=13)
ax.set_ylabel('Latent Feature 2', fontsize=13)
ax.set_zlabel('Latent Feature 3', fontsize=13)

ax.set_xlim3d(0,60)
ax.set_ylim3d(0,60)
ax.set_zlim3d(0,60)

plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=10, bbox_to_anchor=(0, 0))

plt.show()



print("Completed")
