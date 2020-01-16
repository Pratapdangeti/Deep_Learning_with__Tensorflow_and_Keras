
# Regression model using Keras

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import expand_dims

from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import r2_score


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

wine_quality = pd.read_csv(data_path+"winequality-red.csv",sep=';')
# Step for converting white space in columns to _ value for better handling
wine_quality.rename(columns=lambda x: x.replace(" ", "_"), inplace=True)


# Multi linear regression model
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



def DNN_Regression_KS(input_dim):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(input_dim,)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    # Layer 3
    model.add(Dense(1))
    adam_opt = Adam(lr=0.01)
    # Model compilation
    model.compile(loss='mean_squared_error',optimizer=adam_opt)
    return model


model_trn = DNN_Regression_KS(7)
model_trn.fit(x_train,y_train,batch_size=30,nb_epoch=50)

y_train_pred=model_trn.predict(x_train)
print("Train R-squared :",r2_score(y_train,y_train_pred))

y_test_pred=model_trn.predict(x_test)
print("Test R-squared :",r2_score(y_test,y_test_pred))



