

import numpy as np

from sklearn.datasets import load_digits
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data
y = digits.target

# Printing original shape
print("X shape:",X.shape)
print("Y shape:",y.shape)

input_dim = 64
num_classes=10

# Pre-processing data
x_vars_stdscle = StandardScaler().fit_transform(X)
y_ = np.zeros((np.shape(y)[0],num_classes))
y_[np.arange(np.shape(y)[0]),y]=1
x_train,x_test,y_train,y_test = train_test_split(x_vars_stdscle,y_,train_size = 0.7,random_state=42)

# Model building
model = Sequential()
# Layer 1
model.add(Dense(30, input_shape=(input_dim,)))
model.add(Activation('relu'))

# Layer 2
model.add(Dense(20))
model.add(Activation('relu'))

# Layer 3
model.add(Dense(num_classes))
model.add(Activation('softmax'))
print(model.summary())

# Model definition
def Multi_Classification_KS(_input_dim):
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

    adam_opt = Adam(lr=0.01)
    # Model compilation
    _model.compile(loss='categorical_crossentropy',optimizer=adam_opt)
    return _model


# Model training
training_epochs = 50
batch_size = 30

multi_class_model = Multi_Classification_KS(input_dim)
multi_class_model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs)


print("Completed!")

