# Multi Classification model using Keras

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam


from keras.callbacks import CSVLogger,ModelCheckpoint
csv_logger = CSVLogger('mlevel_logfile.log', append=True, separator=';')


digits = load_digits()
X = digits.data
y = digits.target

print ("\nPrinting first digit")
plt.matshow(digits.images[0])
plt.show()


input_dim = 64
num_classes = 10

x_vars_stdscle = StandardScaler().fit_transform(X)
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y_,train_size = 0.7,random_state=42)



def DNN_Multi_Classification_KS(_input_dim):
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

    adam_opt = Adam(lr=0.01)
    # Model compilation
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)

    return model


# Model training
training_epochs = 50
batch_size = 30

# saving model at each iteration and logging the loss into logger on runtime using list
file_name = 'weights-improvement-{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, csv_logger]


multi_class_model = DNN_Multi_Classification_KS(input_dim)
multi_class_model.fit(x_train,y_train,batch_size=batch_size,
                      epochs=training_epochs,verbose=2,validation_split=0.1,callbacks=callbacks_list)

y_train_pred = multi_class_model.predict_classes(x_train)
y_test_pred = multi_class_model.predict_classes(x_test)

y_train_cls = np.argmax(y_train,axis=1)
y_test_cls = np.argmax(y_test,axis=1)


print("Multi Classification Train Confusion matrix :\n",
      confusion_matrix(y_train_cls,y_train_pred))
print("Multi Classification Test Confusion matrix :\n",
      confusion_matrix(y_test_cls,y_test_pred))

print("Multi Classification Train Accuracy : ",
      round(accuracy_score(y_train_cls,y_train_pred),4))
print("Multi Classification Test Accuracy : ",
      round(accuracy_score(y_test_cls,y_test_pred),4))

