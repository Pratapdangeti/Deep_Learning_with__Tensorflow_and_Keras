
from collections import deque
from keras.models import load_model,Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random


data_path = "D:/Book writing/Actual Book/Deep Learning/Codes/Chapter9/Stock_Trading_DRL/"


class Agent:
    def __init__(self,state_size,is_test=False,model_name = ""):
        self.state_size = state_size
        # Action of Buy, Sell and Sit_back
        self.action_size = 3
        # Size of the memory queue
        self.memory = deque(maxlen=1000)
        # Running memory of latest buy and sell information
        self.inventory = []
        # Model name in case if model provided, this is helpful in testing step
        self.model_name = model_name
        # Flag to check if it is train or test
        self.is_test = is_test

        # Initial value in the exploration-exploitation parameter
        # value of 1.0 means agent completely perform exploration,
        # which is non-greedy approach
        self.epsilon = 1.0
        # Min. value of exploration-exploitation parameter
        # 0.01 means with 1% chance agent will move randomly and not
        # based on Q function feedback
        self.epsilon_min = 0.01
        # Decay that regulates the speed at which epsilon diminishes toward
        # the minimum
        self.epsilon_decay = 0.995
        # Discount factor to convert future value into present value
        self.gamma = 0.95

        # Load the given model during testing phase and create new during training phase
        self.model = load_model(data_path+"data/"+model_name) if is_test else self._model()

    # Creating new model during training phase
    def _model(self):
        model = Sequential()
        model.add(Dense(64,input_dim=self.state_size,activation="relu"))
        model.add(Dense(32,activation="relu"))
        model.add(Dense(16,activation="relu"))
        model.add(Dense(self.action_size,activation="linear"))
        model.compile(loss="mse",optimizer=Adam(lr=0.001))
        return model

    # Compute actions for a given state randomly during training
    # and output from model predict during testing
    def act(self,state):
        if not self.is_test and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    # Experience replay let agent to learn generalized way in long-term
    # More efficient use of previous experience
    def expReplay(self,batch_size):
        mini_batch = []
        lngth = len(self.memory)

        for _i in range(lngth-batch_size+1,lngth):
            mini_batch.append(self.memory[_i])

        for state,action,reward,next_state,done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            target_final = self.model.predict(state)
            target_final[0][action] = target
            self.model.fit(state,target_final,epochs=1,verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


