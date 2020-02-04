import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque


class DQN:
    def __init__(self, env):
        self.env = env  # environment for ref shapes in model
        self.memory = deque(maxlen=2000)  # add trials to mem and use rand sample
        self.gamma = 0.95  # future rewards depreciation factor
        # take a random action rather than the one we would predict to be the best in that scenario and decay a each successive trial
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01  # std learning rate param


def create_model(self):
    model = Sequential()
    state_shape = self.env.observation_space.shape
    model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(self.env.action_space.n))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
    return model


def __init__(self, env):
    self.env = env
    self.memory = deque(maxlen=2000)
    self.gamma = 0.95
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.01
    self.tau = 0.05
    # does actual predictions
    self.model = self.create_model()
    # tracks actions we want model to take
    self.target_model = self.create_model()


# add to memory when going through trials
def remember(self, state, action, reward, new_state, done):
    # need to store done phase for updating reward function
    self.memory.append([state, action, reward, new_state, done])


# training phase
def replay(self):
    batch_size = 32
    if len(self.memory) < batch_size:
        return
    # take a random sample from memory storage
    samples = random.sample(self.memory, batch_size)
    # update Q function  as sum of cur reward and expected future reward
    # depricate that by gamma
    # hanlde terminal / non-terminal states
    for sample in samples:
        state, action, reward, new_state, done = sample
        target = self.target_model.predict(state)
        if done:
            target[0][action] = reward
        else:
            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma
        self.model.fit(state, target, epochs=1, verbose=0)
