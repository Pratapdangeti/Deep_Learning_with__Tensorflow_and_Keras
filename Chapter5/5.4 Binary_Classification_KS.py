

# Binary Classification model using Keras

import pandas as pd
from numpy import expand_dims
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from keras.layers import Dense,Activation
from keras.models import Sequential
from keras.optimizers import Adam


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter5/data/"

credit_data = pd.read_csv(data_path+"credit_data.csv")
credit_data['class'] = credit_data['class']-1
# pre-processing
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

# Concatenating all the variables
credit_data_new = pd.concat([dummy_stseca, dummy_ch, dummy_purpose, dummy_savacc, dummy_presc, dummy_perssx,
                             dummy_othdts, dummy_property, dummy_othinstpln, dummy_housing, dummy_job,
                             dummy_telephn, dummy_forgnwrkr, credit_cont_scale_pd, credit_data['class']], axis=1)


# Need to scale the continuous variables and combine for later processing
df_x_train, df_x_test, y_train_inter, y_test_inter = train_test_split(credit_data_new.drop(['class'], axis=1),
                                                    credit_data_new['class'],train_size=0.7, random_state=42)

x_train = df_x_train.as_matrix()
x_test = df_x_test.as_matrix()
y_train = expand_dims(y_train_inter, axis=1)
y_test = expand_dims(y_test_inter, axis=1)

def DNN_Binary_Classification_KS(_input_dim):
    # Layer 1
    model = Sequential()
    model.add(Dense(10,input_shape=(_input_dim,)))
    model.add(Activation('relu'))
    # Layer 2
    model.add(Dense(10))
    model.add(Activation('relu'))
    # Layer 3
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    adam_opt = Adam(lr=0.01)
    # Model compilation
    model.compile(loss='binary_crossentropy',optimizer=adam_opt)

    return model

# Model training
input_dim = 61
training_epochs = 50
batch_size = 30

bin_class_model = DNN_Binary_Classification_KS(input_dim)
bin_class_model.fit(x_train,y_train,batch_size=batch_size,epochs=training_epochs)

y_train_pred = bin_class_model.predict_classes(x_train)
y_test_pred = bin_class_model.predict_classes(x_test)


print("Binary Classification Train Confusion matrix :\n",
      confusion_matrix(y_train,y_train_pred))
print("Binary Classification Test Confusion matrix :\n",
      confusion_matrix(y_test,y_test_pred))

print("Binary Classification Train Accuracy : ",
      round(accuracy_score(y_train,y_train_pred),4))
print("Binary Classification Test Accuracy : ",
      round(accuracy_score(y_test,y_test_pred),4))

