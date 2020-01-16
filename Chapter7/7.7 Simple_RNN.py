


# Simple RNN Implementation
import numpy as np

timsteps = 15
input_dim = 16
hidden_dim = 32
output_dim = 10


# Input
X = np.random.random((timsteps,input_dim))

# State
h_t = np.zeros((hidden_dim,))

# Weights - input, hidden and output
W_xh = np.random.random((hidden_dim,input_dim))
W_hh = np.random.random((hidden_dim,hidden_dim))
W_hy = np.random.random((output_dim,hidden_dim))



def soft_max(x):
    e = np.exp(x - np.max(x))
    if e.ndim ==1:
        return e/np.sum(e,axis=0)
    else:
        return e/np.array([np.sum(e,axis=1)]).T


sequence_outputs = []

# Iterating over timestamps
for x_t in X :
    # Computing new state
    new_h_t = np.tanh(np.dot(W_xh,x_t)+np.dot(W_hh,h_t))
    # Computing output
    output_t = soft_max(np.dot(W_hy,new_h_t))
    # state for next epoch would be new epoch
    h_t = new_h_t
    # Appending outputs
    sequence_outputs.append(output_t)



final_output_sequence = np.concatenate(sequence_outputs,axis=0)


