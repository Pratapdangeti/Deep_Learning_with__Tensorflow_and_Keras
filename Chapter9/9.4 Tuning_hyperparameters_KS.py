

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

from keras.layers import Dense,Activation,Dropout
from keras.models import Sequential
from keras.optimizers import Adam,SGD,Adadelta,Adagrad,RMSprop

digits = load_digits()
X = digits.data
y = digits.target


input_dim = 64
num_classes = 10

x_vars_stdscle = StandardScaler().fit_transform(X)
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y_,train_size = 0.7,random_state=42)

y_train_cls = np.argmax(y_train,axis=1)
y_test_cls = np.argmax(y_test,axis=1)


def DNN_Multi_Classification_KS(_input_dim,_learning_rate):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    # Layer 3
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    adam_opt = Adam(lr=_learning_rate)
    # Model compilation
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)

    return model



# Tuning 1 -  Batch size
"""
training_epochs = 60
learnint_rate = 0.01

batch_size_list = [15,32,64,128,256,512]
for batch_size in batch_size_list:
    multi_class_model = DNN_Multi_Classification_KS(input_dim, _learning_rate=learnint_rate)
    multi_class_model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs,verbose=0)
    y_train_pred = multi_class_model.predict_classes(x_train)
    y_test_pred = multi_class_model.predict_classes(x_test)
    print("Batch size:",batch_size,"Train Accuracy : ",round(accuracy_score(y_train_cls,y_train_pred),4),"Test Accuracy : ",round(accuracy_score(y_test_cls,y_test_pred),4))
    # If we do not delete existing model, next iteration starts from what was trained
    del multi_class_model
"""



# Tuning 2 - Training epoch vs. Learning rate
"""
training_epochs_list = [30,50,100,200,300]
learning_rate_list = [0.1,0.04,0.02,0.01]

best_test_accuracy = 0.0

for training_epochs in training_epochs_list:
    for lr in learning_rate_list:
        multi_class_model = DNN_Multi_Classification_KS(input_dim, _learning_rate=lr)
        multi_class_model.fit(x_train,y_train,batch_size=64,epochs=training_epochs,verbose=0)
        y_train_pred = multi_class_model.predict_classes(x_train)
        y_test_pred = multi_class_model.predict_classes(x_test)
        trn_acc = round(accuracy_score(y_train_cls,y_train_pred),4)
        tst_acc = round(accuracy_score(y_test_cls, y_test_pred), 4)
        if tst_acc > best_test_accuracy:
            print("Training epochs:",training_epochs,"lr:",lr,"Train Accuracy : ",trn_acc,"Test Accuracy : ",tst_acc)
            best_test_accuracy = tst_acc
        # If we do not delete existing model, next iteration starts from what was trained
        del multi_class_model
"""



# Tuning 3 - Optimization methods
"""
def DNN_Multi_Classification_WOPTM_KS(_input_dim,_func):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    # Layer 3
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=_func)
    return model

optimizers_list = [
    SGD(lr=0.04),
    Adadelta(lr=0.01,rho=0.95,epsilon=1e-8),
    Adagrad(lr=0.01,epsilon=1e-08),
    RMSprop(lr=0.04,decay=0.9,epsilon=1e-10),
    Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
]

training_epochs =60

for optimzer in optimizers_list:
    multi_class_model = DNN_Multi_Classification_WOPTM_KS(input_dim,_func=optimzer)
    multi_class_model.fit(x_train, y_train, batch_size=64, epochs=training_epochs, verbose=0)
    y_train_pred = multi_class_model.predict_classes(x_train)
    y_test_pred = multi_class_model.predict_classes(x_test)
    trn_acc = round(accuracy_score(y_train_cls, y_train_pred), 4)
    tst_acc = round(accuracy_score(y_test_cls, y_test_pred), 4)
    print("Optimizer: ",optimzer.__class__,  ", Train Accuracy : ", trn_acc, ", Test Accuracy : ", tst_acc)
    # If we do not delete existing model, next iteration starts from what was trained
    del multi_class_model
"""



# Tuning 4 - dropout rate
"""
def DNN_Multi_Classification_WDOPT_KS(_input_dim,_drprate):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(_drprate))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(_drprate))
    # Layer 3
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    adam_opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)
    return model

training_epochs = 60
dropout_rate_list = [0.1,0.3,0.5,0.8]

for drpt in dropout_rate_list:
    multi_class_model = DNN_Multi_Classification_WDOPT_KS(input_dim,_drprate=drpt)
    multi_class_model.fit(x_train, y_train, batch_size=64, epochs=training_epochs, verbose=0)
    y_train_pred = multi_class_model.predict_classes(x_train)
    y_test_pred = multi_class_model.predict_classes(x_test)
    trn_acc = round(accuracy_score(y_train_cls, y_train_pred), 4)
    tst_acc = round(accuracy_score(y_test_cls, y_test_pred), 4)
    print("Dropout rate:",drpt,", Train Accuracy : ",trn_acc,", Test Accuracy : ",tst_acc)
    # If we do not delete existing model, next iteration starts from what was trained
    del multi_class_model
"""



# Tuning 5 - Activation functions
"""
batch_size = 64
training_epochs = 50
learning_rate = 0.04

def DNN_Multi_Classification_WAF_KS(_input_dim,_activ_func):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation(_activ_func))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation(_activ_func))
    # Layer 3
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    adam_opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)
    return model


act_func_list = ['relu','sigmoid','tanh']

for actv in act_func_list:
    multi_class_model = DNN_Multi_Classification_WAF_KS(input_dim,_activ_func =actv)
    multi_class_model.fit(x_train, y_train, batch_size=64, epochs=training_epochs, verbose=0)
    y_train_pred = multi_class_model.predict_classes(x_train)
    y_test_pred = multi_class_model.predict_classes(x_test)
    trn_acc = round(accuracy_score(y_train_cls, y_train_pred), 4)
    tst_acc = round(accuracy_score(y_test_cls, y_test_pred), 4)
    print("Active function:",actv,", Train Accuracy : ",trn_acc,", Test Accuracy : ",tst_acc)
    # If we do not delete existing model, next iteration starts from what was trained
    del multi_class_model

"""


# Tuning 6 - Number of layers and neurons in each layer
training_epochs = 50

def DNN_Multi_Class_Nlyrsnrn_KS(_architecture):
    model = Sequential()
    # 1st Layer
    model.add(Dense(_architecture[1],input_shape=(_architecture[0],)))
    model.add(Activation('relu'))
    # All middle layers
    for _i in range(2,len(_architecture)-1):
        model.add(Dense(_architecture[_i]))
        model.add(Activation('relu'))
    # Final layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    adam_opt = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)
    return model

architecture_list = [[input_dim, 10, 10, num_classes],
                     [input_dim,10, 10, 10, num_classes],
                     [input_dim,40, 30, 20, num_classes]]

for _arch in architecture_list:
    multi_class_model = DNN_Multi_Class_Nlyrsnrn_KS(_arch)
    multi_class_model.fit(x_train, y_train, batch_size=64, epochs=training_epochs, verbose=0)
    y_train_pred = multi_class_model.predict_classes(x_train)
    y_test_pred = multi_class_model.predict_classes(x_test)
    trn_acc = round(accuracy_score(y_train_cls, y_train_pred), 4)
    tst_acc = round(accuracy_score(y_test_cls, y_test_pred), 4)
    print("Architecture ",_arch,", Train Accuracy : ",trn_acc,", Test Accuracy : ",tst_acc)
    # If we do not delete existing model, next iteration starts from what was trained
    del multi_class_model







print("Completed!")

