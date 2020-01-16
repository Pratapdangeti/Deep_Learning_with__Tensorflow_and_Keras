
# Binary Classification model using TensorFlow

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

credit_data = pd.read_csv(data_path+"credit_data.csv")
pd.set_option('display.max_columns', 21)
print(credit_data.head())

credit_data['class'] = credit_data['class']-1

dummy_stseca = pd.get_dummies(credit_data['Status_of_existing_checking_account'], prefix='status_exs_accnt')
dummy_ch = pd.get_dummies(credit_data['Credit_history'], prefix='cred_hist')
dummy_purpose = pd.get_dummies(credit_data['Purpose'], prefix='purpose')
dummy_savacc = pd.get_dummies(credit_data['Savings_Account'], prefix='sav_acc')
dummy_presc = pd.get_dummies(credit_data['Present_Employment_since'], prefix='pre_emp_snc')
dummy_perssx = pd.get_dummies(credit_data['Personal_status_and_sex'], prefix='per_stat_sx')
dummy_othdts = pd.get_dummies(credit_data['Other_debtors'], prefix='oth_debtors')
dummy_property = pd.get_dummies(credit_data['Property'], prefix='property')
dummy_othinstpln = pd.get_dummies(credit_data['Other_installment_plans'], prefix='oth_inst_pln')
dummy_housing = pd.get_dummies(credit_data['Housing'], prefix='housing')
dummy_job = pd.get_dummies(credit_data['Job'], prefix='job')
dummy_telephn = pd.get_dummies(credit_data['Telephone'], prefix='telephn')
dummy_forgnwrkr = pd.get_dummies(credit_data['Foreign_worker'], prefix='forgn_wrkr')

continuous_columns = ['Duration_in_month', 'Credit_amount', 'Installment_rate_in_percentage_of_disposable_income',
                      'Present_residence_since', 'Age_in_years', 'Number_of_existing_credits_at_this_bank',
                      'Number_of_People_being_liable_to_provide_maintenance_for']

credit_continuous = credit_data[continuous_columns]

# Scaling continuous variables
scaler = MinMaxScaler()
credit_cont_scale = scaler.fit_transform(credit_continuous.as_matrix())
credit_cont_scale_pd = pd.DataFrame(credit_cont_scale)
credit_cont_scale_pd.columns = continuous_columns

credit_data_new = pd.concat([dummy_stseca, dummy_ch, dummy_purpose, dummy_savacc, dummy_presc, dummy_perssx,
                             dummy_othdts, dummy_property, dummy_othinstpln, dummy_housing, dummy_job,
                             dummy_telephn, dummy_forgnwrkr, credit_cont_scale_pd, credit_data['class']], axis=1)


n_classes = 2

y = credit_data_new['class'].as_matrix()
# Creates one hot encoding by creating number of columns as per class
y_ = np.zeros((np.shape(y)[0],n_classes))
y_[np.arange(np.shape(y)[0]),y]=1

# Need to scale the continuous variables and combine for later processing
df_x_train, df_x_test, y_train, y_test = train_test_split(credit_data_new.drop(['class'], axis=1),
                                                                      y_,train_size=0.7, random_state=42)

x_train = df_x_train.as_matrix()
x_test = df_x_test.as_matrix()

# Network architecture
input_dim = 61
layer_1_neurons = 10
layer_2_neurons = 10
learning_rate = 0.01

# Random uniform weights initializer
weights = {
    'W1':tf.Variable(tf.random_uniform([input_dim,layer_1_neurons]),name="W1"),
    'W2':tf.Variable(tf.random_uniform([layer_1_neurons,layer_2_neurons]),name="W2"),
    'WO':tf.Variable(tf.random_uniform([layer_2_neurons,n_classes]),name="WO")
}
# Random uniform weights initializer
biases = {
    'b1':tf.Variable(tf.zeros([layer_1_neurons]),name="b1"),
    'b2':tf.Variable(tf.zeros([layer_2_neurons]),name="b2"),
    'bo':tf.Variable(tf.zeros([n_classes]),name="bo")
}


def DNN_Binary_Classification_TF(_x, _weights, _biases):
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
ys = tf.placeholder(tf.float32,[None,n_classes],name="output_data")

# Construct model
output = DNN_Binary_Classification_TF(xs,weights,biases)

# Define loss and output
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=ys))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

training_epochs = 50
batch_size = 30

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("/tmp/tf_binclasslogs",sess.graph)
    # Running over epochs
    for epoch in range(training_epochs):
        batch_count = int(x_train.shape[0]/batch_size)
        # Iterate over batches
        for i in range(batch_count):
            batch_x = x_train[(i*batch_size): ((i+1)*batch_size),:]
            batch_y = y_train[(i*batch_size): ((i+1)*batch_size),:]
            # Main optimization
            trc,trs = sess.run([train_op,cost_op],feed_dict={xs:batch_x, ys:batch_y})
            # Printing each epoch and each batch
            print("Epoch :",epoch+1,"batch :",i+1," Train Cost :",sess.run(cost_op,feed_dict={xs:x_train, ys:y_train}),
                  "Test Cost :", sess.run(cost_op, feed_dict={xs: x_test, ys: y_test}))

        trcost = sess.run(cost_op,feed_dict={xs:x_train, ys:y_train})
        tstcost = sess.run(cost_op,feed_dict={xs:x_test, ys:y_test})
        # Writing loss values to tensorboard at each epoch
        tr_summary = tf.Summary(value=[tf.Summary.Value(tag="train loss", simple_value=trcost)])
        writer.add_summary(tr_summary, epoch)
        tst_summary = tf.Summary(value=[tf.Summary.Value(tag="test loss", simple_value=tstcost)])
        writer.add_summary(tst_summary, epoch)

    print("Optimization Finished!")

    act_amax = tf.argmax(ys,1)
    pred_amax = tf.argmax(output, 1)


    print("Binary Classification Train Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train}) ))
    print("Binary Classification Test Confusion matrix :\n",confusion_matrix(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test}) ))

    print("Binary Classification Train Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_train}), pred_amax.eval({xs:x_train})),4))
    print("Binary Classification Test Accuracy : ",round(accuracy_score(act_amax.eval({ys:y_test}), pred_amax.eval({xs:x_test})),4))


























