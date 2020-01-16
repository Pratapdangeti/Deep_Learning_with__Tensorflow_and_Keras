

from Agent_Class import Agent
from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter9/Stock_Trading_DRL/"

stock_name = '^NSEI.csv'
data_input = pd.read_csv(data_path+'/data/'+stock_name)
# Dropping rows with NAN closing price
data_input = data_input.dropna(axis=0, subset=['Close'])

# Checking the shape
print(data_input.shape)
# Taking after 200 data points for testing
data = data_input['Close'][200:].to_list()
data_x = data_input['Date'][200:].to_list()

# Model name to load
model_name = "model_epd100"

# Loading the trained model
model = load_model(data_path+"data/"+model_name)
window_size = model.layers[0].input.shape.as_list()[1]

# Calling Agent class
agent = Agent(state_size=window_size,is_test=True,model_name=model_name)

lngth = len(data)-1
batch_size = 32


# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns n-day state representation ending at time t
def getState(_data, _t, _n):
	d = _t - _n + 1
	block = _data[d:_t + 1] if d >= 0 else -d * [_data[0]] + _data[0:_t + 1]
	reps = []
	for i in range(_n - 1):
		reps.append(sigmoid(block[i + 1] - block[i]))
	return np.array([reps])

# Compute initial state
state = getState(_data=data,_t=0,_n=window_size+1)
total_profit = 0
agent.inventory = []

states_buy_list = []
states_sell_list = []

for _iter in range(lngth):
    action = agent.act(state)

    next_state = getState(data,_iter+1,window_size+1)
    reward = 0

    # Buy
    if action==1:
        agent.inventory.append(data[_iter])
        print("Buy: ", round(data[_iter], 1))
        states_buy_list.append(_iter)

    # Sell
    elif action==2 and len(agent.inventory)>0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[_iter]-bought_price,0)
        current_profit = round(data[_iter]-bought_price,1)
        total_profit += current_profit
        print("Sell: ",round(data[_iter],1),"| Profit:",current_profit)
        states_sell_list.append(_iter)

    if _iter == lngth-1:
        done = True
    else:
        done = False

    # Keep appending to memory for utilizing in experience Replay
    agent.memory.append((state,action,reward,next_state,done))
    state = next_state

    if done:
        print("Total Profit:",round(total_profit,1))


# Plotting on test data
figure = data_path + "data/mdel_test.png"
plt.figure(figsize=(10, 5))
plt.plot(data, label='NIFTY', c='black')
plt.plot(data, 'o', label='predict buy', markevery=states_buy_list, c='g')
plt.plot(data, 'o', label='predict sell', markevery=states_sell_list, c='r')
plt.legend()
plt.title("NIFTY: {} {}, Total profit: {} ".format(data_x[0],data_x[-1],round(total_profit, 1)))
plt.savefig(figure)
plt.show()
plt.close()



print("Completed!")
