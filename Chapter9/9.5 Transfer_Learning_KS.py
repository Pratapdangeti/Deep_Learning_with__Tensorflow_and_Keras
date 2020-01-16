
import numpy as np
from keras.applications import VGG16
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dropout,Dense,BatchNormalization,Flatten
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score


# Data preparation stage
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
y_train_2 = np.zeros((np.shape(y_train)[0], 10))
y_train_2[np.arange(np.shape(y_train)[0]), np.ndarray.flatten(y_train)] = 1

y_test_2 = np.zeros((np.shape(y_test)[0], 10))
y_test_2[np.arange(np.shape(y_test)[0]), np.ndarray.flatten(y_test)] = 1

# Number of data points to keep for training due to memory constraint
num_dpoints = 20000

x_train = x_train[:num_dpoints, :]
y_train_2 = y_train_2[:num_dpoints, :]
print(x_train.shape,y_train_2.shape,x_test.shape,y_test_2.shape)

# Importing pre-trained Model
# Include_top is False for customizing the input shape
# rather than default shape 224 x 224 x 3
model = VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))
print(model.summary())

# Take until block4_conv3 layer only
partial_trained_model = Model(input=model.input,output=model.get_layer('block4_conv3').output)
print(partial_trained_model.summary())

print("Original partial trained model layers trainable :")
for layer in partial_trained_model.layers:
    print(layer.name,layer.trainable)

partial_trained_model.trainable = True
# Indicator for later stages
set_trainable = False

for layer in partial_trained_model.layers:
    if layer.name == 'block4_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False


print("After freezing top layers of  partial \n \
      trained model layers trainable :")
for layer in partial_trained_model.layers:
    print(layer.name,layer.trainable)


# After confirming the layer trainability, we are adding additional layers to
# customize it to CIFAR problem
# Dense Layer 2
x_out_int = partial_trained_model.output
x_out_int = Flatten()(x_out_int)
x_out_int = Dense(128,activation='relu')(x_out_int)
x_out_int = Dropout(0.5)(x_out_int)
x_out_int = BatchNormalization()(x_out_int)

x_out_int = Dropout(0.5)(x_out_int)
preds = Dense(10,activation='softmax')(x_out_int)

# Final model architecture
model_final = Model(input = partial_trained_model.input,output=preds)

adam_opt = Adam(lr=0.01)
# Model compilation
model_final.compile(loss='categorical_crossentropy',optimizer=adam_opt,metrics=['accuracy'])
print(model_final.summary())

print("Final model trainable check :")
for layer in model_final.layers:
    print(layer.name,layer.trainable)

# Model training
training_epochs = 10
batch_size = 128
model_final.fit(x_train,y_train_2,batch_size=batch_size,
                    epochs=training_epochs,verbose=1,validation_split=0.2)

y_train_pred_prob = model_final.predict(x_train)
y_train_pred = y_train_pred_prob.argmax(axis=-1)
y_test_pred_prob = model_final.predict(x_test)
y_test_pred = y_test_pred_prob.argmax(axis=-1)

y_train_cls = np.argmax(y_train_2,axis=1)
y_test_cls = np.argmax(y_test_2,axis=1)

print("\n")
print("CIFAR-10 Classification Train Accuracy : ",round(accuracy_score(y_train_cls,y_train_pred),4))
print("CIFAR-10 Classification Test Accuracy : ",round(accuracy_score(y_test_cls,y_test_pred),4))


print("Completed!")

