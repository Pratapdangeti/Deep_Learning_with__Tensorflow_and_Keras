

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam


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

def DNN_Multi_Classification_KS(_input_dim):
    # Layer 1
    _model = Sequential()
    _model.add(Dense(10,input_shape=(_input_dim,)))
    _model.add(Activation('relu'))
    # Layer 2
    _model.add(Dense(10))
    _model.add(Activation('relu'))
    # Layer 3
    _model.add(Dense(num_classes))
    _model.add(Activation('softmax'))
    adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Model compilation
    _model.compile(loss='categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy'])
    return _model

# Training parameters
training_epochs = 60
batch_size = 64

model = DNN_Multi_Classification_KS(input_dim)

# Training model
model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1)

y_train_pred = model.predict_classes(x_train)
y_test_pred = model.predict_classes(x_test)

# Comparing the results for checking purpose
print("Original Model results ","Train Accuracy : ", round(accuracy_score(y_train_cls, y_train_pred), 4),
      "Test Accuracy : ", round(accuracy_score(y_test_cls, y_test_pred), 4))


# Using generators for data preparation
def data_Generator(_x_data,_y_data,_batch_size =32):
    while True:
        num_batches = int(np.ceil(_x_data.shape[0]/_batch_size))
        for _i in range(num_batches):
            yield _x_data[(_i*_batch_size):((_i+1)*_batch_size)],\
                  _y_data[(_i*_batch_size):((_i+1)*_batch_size)]


# Using model generator to train on generated data
model.fit_generator(data_Generator(x_train,y_train,_batch_size=batch_size),validation_data=(x_test,y_test),
                    steps_per_epoch=int(np.ceil(x_train.shape[0]/batch_size)),epochs=training_epochs)

y_train_pred_gen = model.predict_generator(data_Generator(x_train,y_train,_batch_size=batch_size),
                                           steps=int(np.ceil(x_train.shape[0]/batch_size)))

y_test_pred_gen = model.predict_generator(data_Generator(x_test,y_test,_batch_size=batch_size),
                                           steps=int(np.ceil(x_test.shape[0]/batch_size)))

y_train_pred_gen_cls = np.argmax(y_train_pred_gen,axis=1)
y_test_pred_gen_cls = np.argmax(y_test_pred_gen,axis=1)

# Comparing the results for checking purpose
print("Generator Model results ","Train Accuracy : ", round(accuracy_score(y_train_cls, y_train_pred_gen_cls), 4),
      "Test Accuracy : ", round(accuracy_score(y_test_cls, y_test_pred_gen_cls), 4))


print("Completed!")
