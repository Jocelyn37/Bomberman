import pickle
import random
from collections import namedtuple, deque
from typing import List
import numpy as np
import os

import events as e
from sklearn.multioutput import MultiOutputRegressor as MOR

from sys import path
path.append(r"/Users/jocelyn/LightGBM/python-package")
from lightgbm import LGBMRegressor as LGBMR
from random import shuffle
from sklearn.ensemble import GradientBoostingRegressor



ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']
n_actions = len(ACTIONS)


# hyperparameters
alpha = 0.75 # learning rate
gama = 0.95 # discount


# features [up, right, down, left, nearest coin(coordinate x), nearest coin(coordinate y)]
n_features = 6 #4


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    
    self.last_actions = np.zeros((0,1),dtype=np.int8) # empty at present
    #self.last_actions = np.vstack((self.last_actions, np.random.randint(len(ACTIONS))))
    # model construction
    self.Q_values = np.zeros((0,n_actions))
    self.states = np.zeros((0,n_features))
    self.rewards = np.zeros((0,n_actions))
    
def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.logger.debug(f'EVENTS: {events}')
    if len(events) != 0:
        # update the rewards list
        current_reward = np.zeros((1, n_actions))
        current_reward[0][self.last_actions[-1]] = reward_from_events(self, events)#r
        self.rewards = np.vstack((self.rewards, current_reward))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    
    
    # update the rewards list
    current_reward = np.zeros((1, n_actions))
    current_reward[0][self.last_actions[-1]] = reward_from_events(self, events)#r
    self.rewards = np.vstack((self.rewards, current_reward))
    
    # update rule
    self.Q_values = np.zeros((0,n_actions))
    if os.path.isfile("states.csv"):
        q_vs = np.amax(self.regression.predict(self.states), axis=1) # predict Q(s,a)
        for i in range(len(q_vs) - 1):
            a = self.last_actions[i][0]
            R = self.rewards[i][a] # current reward
            TD = R + gama * q_vs[i+1] - q_vs[i] # Temporal difference of Q-learning
            Q_s_a = np.zeros((1,n_actions)) # new Q(s,a)
            Q_s_a[0][a] = q_vs[i] + alpha * TD
            self.Q_values = np.vstack((self.Q_values, Q_s_a))
        self.Q_values = np.vstack((self.Q_values, self.rewards[i+1]))
        #print("***",i,"***")
        
    else:
        self.Q_values = self.rewards

    self.regression = MOR(GradientBoostingRegressor())
    # self.regression = MOR(GradientBoostingRegressor()) # NAN or zeros are treated as missing
    self.regression.fit(self.states,self.Q_values)
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Store the model
    np.savetxt('states.csv', self.states, delimiter=',')
    np.savetxt('Q_values.csv', self.Q_values, delimiter=',')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is to modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.INVALID_ACTION: -10,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        else:
            reward_sum -=1
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
