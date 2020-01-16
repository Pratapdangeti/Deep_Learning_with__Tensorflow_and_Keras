

from Agent_Class import Agent

data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter9/Stock_Trading_DRL/"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

stock_name = '^NSEI.csv'

data_input = pd.read_csv(data_path+'/data/'+stock_name)
# Dropping rows with NAN closing price
data_input = data_input.dropna(axis=0, subset=['Close'])

# Checking the shape
print(data_input.shape)
# Taking first 200 data points for training
data = data_input['Close'][:200].to_list()
data_x = data_input['Date'][:200].to_list()


# Parameters
window_size = 10
episode_count = 100
batch_size = 32

agent = Agent(state_size = window_size)
lngth = len(data)-1

states_buy_list = []
states_sell_list = []


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

# Iterating over total episodes
for epd in range(episode_count+1):
    print("Episode:",epd,"/",episode_count)
    state = getState(_data=data,_t=0,_n=window_size+1)

    total_profit = 0
    agent.inventory = []

    # Iterating over data points
    for _iter in range(lngth):
        action = agent.act(state)

        next_state = getState(_data=data,_t=_iter+1,_n=window_size+1)
        reward = 0

        # Buy
        if action == 1:
            agent.inventory.append(data[_iter])
            print("Buy: ",round(data[_iter],1))
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

        # Use experiece replay whenever memory have more than batch size value
        if len(agent.memory)>batch_size:
            agent.expReplay(batch_size)

    # Save model and plot for every 10 epochs
    if epd%10 == 0:
        agent.model.save(data_path+"data/model_epd"+str(epd))
        # Plotting for every 10th episode
        figure = data_path + "data/mdel_epd_{}_chart.png".format(str(epd))
        plt.figure(figsize=(10, 5))
        plt.plot(data, label='NIFTY', c='black')
        plt.plot(data, 'o', label='predict buy', markevery=states_buy_list, c='g')
        plt.plot(data, 'o', label='predict sell', markevery=states_sell_list, c='r')
        plt.legend()
        plt.title("NIFTY: {} {},  Episode:{}, Total profit: {} ".format(data_x[0],data_x[-1],str(epd), round(total_profit, 1)))
        plt.savefig(figure)
        plt.show()
        plt.close()





print("Completed!")
