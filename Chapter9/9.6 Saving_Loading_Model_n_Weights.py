


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter9/"

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
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    # Layer 3
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    adam_opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Model compilation
    model.compile(loss='categorical_crossentropy',optimizer=adam_opt)
    return model

# Training parameters
training_epochs = 60
batch_size = 64

original_model = DNN_Multi_Classification_KS(input_dim)
original_model.fit(x_train, y_train, batch_size=batch_size, epochs=training_epochs, verbose=1)

y_train_pred = original_model.predict_classes(x_train)
y_test_pred = original_model.predict_classes(x_test)

# Serialize model to JSON and saving
model_json = original_model.to_json()
with open(data_path+'data/original_model.json', "w") as json_file:
    json_file.write(model_json)

# Saving the model weights
original_model.save_weights(data_path+'data/original_model_weights.h5')

# Loading the model and weights
from keras.models import model_from_json

# Loading the model
inp_json_file = open(data_path+'data/original_model.json','r')
loaded_model_json = inp_json_file.read()
inp_json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Loading model weights
loaded_model.load_weights(data_path+'data/original_model_weights.h5')

# Just predicting with model without compiling
y_train_pred_loaded = loaded_model.predict_classes(x_train)
y_test_pred_loaded = loaded_model.predict_classes(x_test)

# Comparing the results for checking purpose
print("Original model results:","Train Accuracy : ", round(accuracy_score(y_train_cls, y_train_pred), 4),
      "Test Accuracy : ", round(accuracy_score(y_test_cls, y_test_pred), 4))

print("Loaded model results:","Train Accuracy : ", round(accuracy_score(y_train_cls, y_train_pred_loaded),4),
      "Test Accuracy : ", round(accuracy_score(y_test_cls, y_test_pred_loaded),4))



print("Completed")
