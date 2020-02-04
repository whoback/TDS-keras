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
