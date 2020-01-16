
import numpy as np
import matplotlib.pyplot as plt

x_vals = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))


print("Sigmoid :",sigmoid(x_vals))
print("Sigmoid derivative :",sigmoid_derivative(x_vals))

sigmoid_x_vals = sigmoid(x_vals)
sigmoid_derivative_x_vals = sigmoid_derivative(x_vals)

plt.plot(x_vals,sigmoid_x_vals,label='sigmoid')
plt.scatter(x_vals,sigmoid_derivative_x_vals,color='r',label='sigmoid derivative')
plt.xlabel('x-values')
plt.ylabel('function value')
plt.legend(loc='upper left')
plt.show()
plt.close()


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1-(tanh(x)**2)


print("Tanh :",tanh(x_vals))
print("Tanh derivative",tanh_derivative(x_vals))

tanh_x_vals = tanh(x_vals)
tanh_derivative_x_vals = tanh_derivative(x_vals)

plt.plot(x_vals,tanh_x_vals,label='tanh')
plt.scatter(x_vals,tanh_derivative_x_vals,color='r',label='tanh derivative')
plt.xlabel('x-values')
plt.ylabel('function value')
plt.legend(loc='upper left')
plt.show()
plt.close()


def relu(x):
    return x*(x > 0)

def relu_derivative(x):
    return 1*(x > 0)

print("Relu :",relu(x_vals))
print("Relu derivative :",relu_derivative(x_vals))

relu_x_vals = relu(x_vals)
relu_derivative_x_vals = relu_derivative(x_vals)

plt.plot(x_vals,relu_x_vals,label='relu')
plt.scatter(x_vals,relu_derivative_x_vals,color='r',label='relu derivative')
plt.xlabel('x-values')
plt.ylabel('function value')
plt.legend(loc='upper left')
plt.show()
plt.close()


def leaky_relu(x,alpha = 0.01):
    output = np.where(x>0,x,alpha*x)
    return output

def leaky_relu_derivative(x,alpha = 0.01):
    output = np.where(x>0,1,alpha)
    return output

print("Leaky relu : ",leaky_relu(x_vals,alpha=0.1))
print("Leaky relu derivative :",leaky_relu_derivative(x_vals,0.1))


leaky_relu_x_vals = leaky_relu(x_vals,0.1)
leaky_relu_derivative_x_vals = leaky_relu_derivative(x_vals,0.1)


plt.plot(x_vals,leaky_relu_x_vals,label='leaky relu')
plt.scatter(x_vals,leaky_relu_derivative_x_vals,color='r',label='leaky relu derivative')
plt.xlabel('x-values')
plt.ylabel('function value')
plt.legend(loc='upper left')
plt.show()
plt.close()



def soft_max(x):
    e = np.exp(x - np.max(x))
    if e.ndim ==1:
        return e/np.sum(e,axis=0)
    else:
        return e/np.array([np.sum(e,axis=1)]).T


print(soft_max(np.array([1,3,7])))


def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else:
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m

print(softmax_grad(soft_max(np.array([1,3,7]))))



def softmax_grad_vect(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

print(softmax_grad_vect(soft_max(np.array([1,3,7]))))


def linear(x):
    return x


def linear_derivative(x):
    return np.ones(np.shape(x))


print("Linear activation function :",linear(x_vals))
print("Derivative of Linear function :",linear_derivative(x_vals))

linear_x_vals = linear(x_vals)
linear_derivative_x_vals = linear_derivative(x_vals)

plt.plot(x_vals,linear_x_vals,label='linear')
plt.scatter(x_vals,linear_derivative_x_vals,color='r',label='linear derivative')
plt.xlabel('x-values')
plt.ylabel('function value')
plt.legend(loc='upper left')
plt.show()
plt.close()





