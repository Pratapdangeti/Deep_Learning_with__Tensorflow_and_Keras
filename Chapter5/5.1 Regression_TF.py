
# Regression model using TensorFlow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import expand_dims
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import time

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

wine_quality = pd.read_csv(data_path+"winequality-red.csv",sep=';')
# Step for converting white space in columns to _ value for better handling
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)

# Plots - pair plots
eda_colnms = [ 'sulphates','chlorides', 'alcohol','quality']
sns.set(style='whitegrid',context = 'notebook')
sns.pairplot(wine_quality[eda_colnms],size = 2.5,x_vars= eda_colnms,y_vars=eda_colnms)
plt.show()

# Correlation coefficients
corr_mat = np.corrcoef(wine_quality[eda_colnms].values.T)
sns.set(font_scale=1)
full_mat = sns.heatmap(corr_mat, cbar=True, annot=True, square=True, fmt='.2f',
                       annot_kws={'size': 15}, yticklabels=eda_colnms, xticklabels=eda_colnms)
plt.show()


colnms = ['volatile_acidity','chlorides','free_sulfur_dioxide','total_sulfur_dioxide',
 'pH', 'sulphates', 'alcohol']

pdx = wine_quality[colnms]
pdy = wine_quality["quality"]

df_x_train,df_x_test,df_y_train,df_y_test = train_test_split(pdx,pdy,train_size = 0.7,random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(df_x_train.as_matrix())
x_test = scaler.fit_transform(df_x_test.as_matrix())

y_train_inter = df_y_train.as_matrix()
y_test_inter = df_y_test.as_matrix()


y_train = expand_dims(y_train_inter, axis=1)
y_test = expand_dims(y_test_inter,axis=1)


def DNN_Regression_TF(X_data, input_dim):

    # Layer 1
    layer_1_neurons = 10
    # Initializing random weights for hidden layer 1
    W1 = tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]),name="W1")
    # Initializing bias vector for hidden layer 1
    b1 = tf.Variable(tf.zeros([layer_1_neurons]),name="b1")
    # Multiplying and adding bias
    Y1 = tf.add(tf.matmul(X_data,W1),b1,name="Y1")
    # Applying activation
    Y1 = tf.nn.relu(Y1)

    # Layer 2
    layer_2_neurons = 10
    W2 = tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]),name="W2")
    b2 = tf.Variable(tf.zeros([layer_2_neurons]),name="b2")
    Y2 = tf.add(tf.matmul(Y1,W2),b2,name="Y2")
    Y2 = tf.nn.relu(Y2)

    # Layer 3
    layer_3_neurons = 1
    WO = tf.Variable(tf.random_uniform([layer_2_neurons,layer_3_neurons]),name="WO")
    bo = tf.Variable(tf.zeros([layer_3_neurons]),name="bo")
    output_layer = tf.add(tf.matmul(Y2,WO),bo,name="OL")

    return output_layer


# placeholder for variables
xs = tf.placeholder(tf.float32,[None,7],name="Input_data")
ys = tf.placeholder(tf.float32,[None,1],name="output_data")

output = DNN_Regression_TF(xs,7)
cost_op = tf.reduce_mean(tf.square(output-ys))

# train model
#train_op =tf.train.GradientDescentOptimizer(0.001).minimize(cost_op)
train_op =tf.train.AdamOptimizer(0.01).minimize(cost_op)

# Create a summary to monitor cost tensor
tf.summary.scalar("cost", cost_op)

# Create a summary to monitor accuracy tensor
summary_op = tf.summary.merge_all()

training_epochs = 50
batch_size = 30

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("/tmp/tf_regression_logs",sess.graph)
    start = time.time()
    for epoch in range(training_epochs):
        batch_count = int(x_train.shape[0]/batch_size)
        for i in range(batch_count):
            batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
            batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]

            trc,trs = sess.run([train_op,summary_op],feed_dict={xs:batch_x, ys:batch_y})

            print("Epoch :",epoch,"batch :",i," Train Cost :",sess.run(cost_op,feed_dict={xs:x_train, ys:y_train}),
                  "Test Cost :", sess.run(cost_op, feed_dict={xs: x_test, ys: y_test}))

        trcost = sess.run(cost_op,feed_dict={xs:x_train, ys:y_train})
        tstcost = sess.run(cost_op,feed_dict={xs:x_test, ys:y_test})

        # Writing loss values to tensorboard at each epoch
        tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=trcost)])
        writer.add_summary(tr_summary, epoch)
        tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=tstcost)])
        writer.add_summary(tst_summary, epoch)

    print("Optimization finished!")

    # Calculating r-square value for evaluating effectiveness of model
    y_train_pred = sess.run(output,feed_dict={xs:x_train})
    print("Regression Train R-squared : ",r2_score(y_train,y_train_pred))

    y_test_pred = sess.run(output,feed_dict={xs:x_test})
    print("Regression Test R-squared : ",r2_score(y_test,y_test_pred))

    end = time.time()
    print(end-start)





