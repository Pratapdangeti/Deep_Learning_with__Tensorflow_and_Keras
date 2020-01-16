

import numpy as np
# Method 2 : Use Keras inbuilt functions
from keras.datasets import cifar10

from keras.layers import Dense,Activation,Conv2D,\
    MaxPool2D,BatchNormalization,Flatten,Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,confusion_matrix

import time

# Data preparation stage
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train_2 = np.zeros((np.shape(y_train)[0], 10))
y_train_2[np.arange(np.shape(y_train)[0]), np.ndarray.flatten(y_train)] = 1

y_test_2 = np.zeros((np.shape(y_test)[0], 10))
y_test_2[np.arange(np.shape(y_test)[0]), np.ndarray.flatten(y_test)] = 1

# Number of data points to keep for training due to memory constraint
num_dpoints = 20000

x_train = x_train[:num_dpoints, :]
y_train_2 = y_train_2[:num_dpoints, :]

print(x_train.shape,y_train_2.shape,x_test.shape,y_test_2.shape)


def CNN_Classification_KS(_keep_prob):

    model = Sequential()
    # Layer 1
    model.add(Conv2D(64,(5,5),padding='same',input_shape=(32,32,3) ) )
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    # Layer 2
    model.add(Conv2D(128,(5,5),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    # Layer 3
    model.add(Conv2D(256,(3,3),padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    # Flattening Layer
    model.add(Flatten())
    # Dense Layer 1
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(1-_keep_prob))
    model.add(BatchNormalization())
    # Dense Layer 2
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(1-_keep_prob))
    model.add(BatchNormalization())
    # Final output Layer
    model.add(Dropout(1-_keep_prob))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    adam_opt = Adam(lr=0.01)
    # Model compilation
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy'])
    return model


# Model training
training_epochs = 10
batch_size = 512

cnn_class_model = CNN_Classification_KS(_keep_prob=0.5)

start_time = time.time()

cnn_class_model.fit(x_train,y_train_2,batch_size=batch_size,
                    epochs=training_epochs,verbose=1,validation_split=0.2)
end_time = time.time()

print("Seconds : ",round(end_time-start_time,4))


y_train_pred = cnn_class_model.predict_classes(x_train)
y_test_pred = cnn_class_model.predict_classes(x_test)

y_train_cls = np.argmax(y_train_2,axis=1)
y_test_cls = np.argmax(y_test_2,axis=1)

print("Train Confusion :\n")
print("CIFAR-10 Classification Train Confusion matrix :\n",confusion_matrix(y_train_cls,y_train_pred))
print("Test Confusion :\n")
print("CIFAR-10 Classification Test Confusion matrix :\n", confusion_matrix(y_test_cls,y_test_pred))

print("Completed!\n")
print("CIFAR-10 Classification Train Accuracy : ",round(accuracy_score(y_train_cls,y_train_pred),4))
print("CIFAR-10 Classification Test Accuracy : ",round(accuracy_score(y_test_cls,y_test_pred),4))



