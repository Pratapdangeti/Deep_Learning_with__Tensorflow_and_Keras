
import tensorflow as tf
import numpy as np
import random
from numpy import expand_dims
import matplotlib.pyplot as plt

random.seed(1024 * 1024)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshaping columns from 28 x 28 into 784
x_train_2 = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test_2 = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

# Normalization
x_train_mean = x_train_2.mean(0)
x_train_std = x_train_2.std(0)
x_train_2 = 1.0 * (x_train_2 - x_train_mean) / (x_train_std + 0.00001)
x_test_2 = 1.0 * (x_test_2 - x_train_mean) / (x_train_std + 0.00001)


sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

class NeuralNetwork:

    def __init__(self,num_input,num_hidden,num_output):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.num_params = self.num_hidden * (self.num_input + 1) + self.num_output * (self.num_hidden + 1)
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))

    def gradient (self,x,y,_lambda,_w):
        input_len = len(x)
        w1 = _w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        w2 = _w[self.num_hidden * (self.num_input + 1): ].reshape(self.num_output, self.num_hidden + 1)

        b = np.matrix(np.ones([input_len, 1]))
        a1 = np.column_stack([x, b])
        s2 = sigmoid(a1 * w1.T)

        a2 = np.column_stack([s2, b])
        a3 = sigmoid(a2 * w2.T)

        y_one_hot = np.matrix(np.zeros([input_len, self.num_output]))
        y_one_hot[(np.matrix(range(input_len)), y.T)] = 1

        #cost = (1.0 / input_len) * (- np.multiply(y_one_hot, np.log(a3)) - np.multiply(1.0 - y_one_hot, np.log(1.0 - a3))).sum()
        cost = (1.0 / input_len) * (- np.multiply(y_one_hot, np.log(a3+1e-15)) ).sum()

        cost += (_lambda / (2.0 * input_len)) * (np.square(w1[:, 0: -1]).sum() + np.square(w2[:, 0: -1]).sum())

        delta3 = a3 - y_one_hot
        delta2 = np.multiply(delta3 * w2[:, 0: -1], np.multiply(s2, 1.0 - s2))

        l1_grad = delta2.T * a1
        l2_grad = delta3.T * a2

        r1_grad = np.column_stack([w1[:, 0: -1], np.matrix(np.zeros([self.num_hidden, 1]))])
        r2_grad = np.column_stack([w2[:, 0: -1], np.matrix(np.zeros([self.num_output, 1]))])

        w1_grad = (1.0 / input_len) * l1_grad + (1.0 * _lambda / input_len) * r1_grad
        w2_grad = (1.0 / input_len) * l2_grad + (1.0 * _lambda / input_len) * r2_grad
        w_grad = np.row_stack([w1_grad.reshape(-1, 1), w2_grad.reshape(-1, 1)])

        return cost, w_grad


    def train_sgd(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,_eta = 0.05):
        print("SGD Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                # Updating weights with gradients
                w = w - _eta * grad
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch


    def train_momentum(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,_eta = 0.05,_gamma=0.9):
        print("Momentum Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            v = np.matrix(np.zeros(w.shape))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                # Momentum updation
                v = _gamma * v + _eta * grad
                w = w - v
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch

    def train_nag(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,_eta = 0.05,_gamma=0.9):
        print("NAG Momentum Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            v = np.matrix(np.zeros(w.shape))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w -_gamma*v)
                # NAG Momentum updation
                v = _gamma * v + _eta * grad
                w = w - v
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch


    def train_adagrad(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,_eta = 0.05,_epsilon = 1e-8):
        print("Adagrad Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        grad_sum_square = np.matrix(np.zeros(w.shape))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                # Adaptive gradients
                grad_sum_square += np.square(grad)
                delta = - _eta * grad / np.sqrt(grad_sum_square+_epsilon)
                w = w + delta
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch


    def train_Adadelta(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,
                       _gamma=0.9,_eta = 0.05,_epsilon = 1e-8):
        print("Adadelta Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        # adadelta parameters
        grad_expect = np.matrix(np.zeros(w.shape))
        delta_expect = np.matrix(np.zeros(w.shape))
        first_run = True
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                grad_expect = _gamma * grad_expect + (1.0 -_gamma)*np.square(grad)
                if first_run == True:
                    delta = - _eta * grad
                else:
                    delta = -np.multiply( np.sqrt(delta_expect + _epsilon)/np.sqrt(grad_expect+_epsilon) ,grad)
                w = w + delta
                delta_expect = _gamma * delta_expect + (1.0-_gamma)* np.square(delta)
                i += batch_size
                cost_array = np.append(cost_array,cost)
                if first_run == True:
                    first_run = False
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch


    def train_RMSprop(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,
                       _gamma=0.9,_eta = 0.05,_epsilon = 1e-8):
        print("RMSprop Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        mean_cost_epoch = np.empty((num_epoch,2))
        # RMSprop parameters
        grad_expect = np.matrix(np.zeros(w.shape))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                grad_expect = _gamma * grad_expect + (1.0 -_gamma)*np.square(grad)
                w = w - _eta * grad /np.sqrt(grad_expect + _epsilon)
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
        self.w = w
        self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
        self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch


    def train_Adam(self,x,y,num_epoch = 100, batch_size =3000,_lambda=0.01,
                       _eta = 0.05,_epsilon = 1e-8, _beta1=0.9, _beta2 =0.999,
                   _beta1_exp=1.0,_beta2_exp=1.0):
        print("Adam Optimization start")
        self.w = np.matrix(0.005 * np.random.random([self.num_params, 1]))
        data = np.hstack((x,expand_dims(y, axis=1)))
        w = self.w
        # Adam parameters
        v = np.matrix(np.zeros(w.shape))
        m = np.matrix(np.zeros(w.shape))
        mean_cost_epoch = np.empty((num_epoch,2))
        for epoch in range(num_epoch):
            np.random.shuffle(data)
            i = 0
            cost_array = np.empty((0,0))
            while i < len(data):
                x = data[i: i + batch_size, 0: -1]
                y = np.matrix(data[i: i + batch_size, -1], dtype='int32')
                cost, grad = self.gradient(x, y, _lambda,w)
                m = _beta1 * m + (1.0-_beta1)*grad
                v = _beta2 * v + (1.0-_beta2)*np.square(grad)
                _beta1_exp *= _beta1
                _beta2_exp *= _beta2
                w = w- _eta * (m/(1.0-_beta1_exp))/(np.sqrt(v/(1.0-_beta2_exp))+_epsilon)
                i += batch_size
                cost_array = np.append(cost_array,cost)
            mean_cost_epoch[epoch,0] = epoch+1
            mean_cost_epoch[epoch,1] = np.mean(cost_array)
            print("Epoch:",epoch+1, ", Cost:", np.mean(cost_array))
            epoch += 1
            self.w = w
            self.w1 = w[0: self.num_hidden * (self.num_input + 1)].reshape(self.num_hidden, self.num_input + 1)
            self.w2 = w[self.num_hidden * (self.num_input + 1):].reshape(self.num_output, self.num_hidden + 1)
        return mean_cost_epoch

    def predict(self, x):
        num_sample = len(x)
        b = np.matrix(np.ones([num_sample, 1]))
        h1 = sigmoid(np.column_stack([x, b]) * self.w1.T)
        h2 = sigmoid(np.column_stack([h1, b]) * self.w2.T)
        return np.argmax(h2, 1)

    def accuracy_test(self, x, y):
        num_sample = len(x)
        y_pred = self.predict(x)
        y_one_hot = np.zeros((y.shape[0],1))
        #y_one_hot = np.matrix(np.zeros(y.shape[0], 1))
        y_one_hot[np.where(y_pred == y)] = 1
        acc = 1.0 * y_one_hot.sum() / num_sample
        return acc




# Model training code
num_input = 28 * 28
num_hidden = 25
num_output = 10

model = NeuralNetwork(num_input ,num_hidden, num_output)



# SGD Optimization
cost_epoch_sgd = model.train_sgd(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_eta=0.05)

plt.plot(cost_epoch_sgd[:,0],cost_epoch_sgd[:,1])
plt.title("SGD Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()


"""
# Momentum Optimization
cost_epoch_momentum = model.train_momentum(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_eta=0.05,_gamma=0.9)

plt.plot(cost_epoch_momentum[:,0],cost_epoch_momentum[:,1])
plt.title("Momentum Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()
"""

"""
# NAG Momentum Optimization
cost_epoch_nag = model.train_nag(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_eta=0.05,_gamma=0.9)

plt.plot(cost_epoch_nag[:,0],cost_epoch_nag[:,1])
plt.title("NAG Momentum Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()
"""

"""
# Adagrad  Optimization
cost_epoch_adagrad = model.train_adagrad(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_eta=0.05,_epsilon = 1e-8)

plt.plot(cost_epoch_adagrad[:,0],cost_epoch_adagrad[:,1])
plt.title("Adagrad Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()
"""

"""
# Adadelta  Optimization
cost_epoch_Adadelta = model.train_Adadelta(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_gamma=0.9,_eta=0.05,_epsilon = 1e-8)

plt.plot(cost_epoch_Adadelta[:,0],cost_epoch_Adadelta[:,1])
plt.title("Adadelta Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()
"""


"""
# RMSprop  Optimization
cost_epoch_RMSprop = model.train_RMSprop(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_gamma=0.9,_eta=0.001,_epsilon = 1e-8)

plt.plot(cost_epoch_RMSprop[:,0],cost_epoch_RMSprop[:,1])
plt.title("RMSProp Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()
"""
"""

# Adam  Optimization
cost_epoch_Adam = model.train_Adam(x_train_2,y_train,num_epoch=100,
                batch_size=60,_lambda=0.01,_eta=0.001,_epsilon = 1e-8,
                _beta1=0.9, _beta2=0.999, _beta1_exp=1.0, _beta2_exp=1.0)

plt.plot(cost_epoch_Adam[:,0],cost_epoch_Adam[:,1])
plt.title("Adam Optimization")
plt.xlabel("Number of epochs")
plt.ylabel("Mean Cross entropy error")
plt.ylim(0,0.05)
plt.close()

"""



print("completed")



