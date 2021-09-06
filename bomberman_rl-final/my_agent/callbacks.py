import os
import pickle
import random
from random import shuffle
import numpy as np
from sys import path
path.append(r"/Users/jocelyn/LightGBM/python-package")
from lightgbm import LGBMRegressor as LGBMR
from sklearn.multioutput import MultiOutputRegressor as MOR

from sklearn.ensemble import GradientBoostingRegressor

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN']



def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards closest target or towards tile closest to any target.
    """
    if len(targets) == 0: return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]
    
        

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("states.csv"):
        self.logger.info("Setting up model from scratch.")
    else:
        self.logger.info("Loading model from saved state.")
        self.states = np.loadtxt('states.csv', delimiter = ',')
        self.Q_values = np.loadtxt('Q_values.csv', delimiter = ',')
        # self.regression = MOR(GradientBoostingRegressor())
        self.regression = MOR(GradientBoostingRegressor())
        self.regression.fit(self.states,self.Q_values) # cause errors with different shape
    np.random.seed()


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    # game_state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']

    # find coordinates of the nearest coin
    free_space = (arena == 0)
    target = []
    target.append(look_for_targets(free_space, (x,y), coins))

    # adjacent fields
    left = arena[x-1, y]
    right = arena[x+1, y]
    up = arena[x, y+1]
    down = arena[x, y-1]
    state = [left, right, up, down]

    # find relative coordinates of targets or (0,0) if target doesn't exist
    for t in target:
        if t != None:
            t = (t[0]-x, t[1]-y)
        else:
            t = (0,0)
        state.append(t[0])
        state.append(t[1])

    # # game_state
    # arena = game_state['field']
    # # _, score, bombs_left, (x, y) = game_state['self']
    # coins = game_state['coins']
    # px, py = game_state['self'][3]
    # # find coordinates of the nearest coin
    # free_space = (arena == 0)
    # target = look_for_targets(free_space, (px, py), coins)
    #
    # dx = np.abs(target[0] - px)
    # dy = np.abs(target[1] - py)
    #
    # state = [px, py, dx, dy]

    # add current state into states
    self.states = np.vstack((self.states, np.asarray(state)))
    
    # to do Exploration vs exploitation
    random_prob = .65
    if self.train:
        np.random.seed()
        if random.random() < random_prob:
            self.logger.debug("Choosing action purely at random.")
            action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
            self.last_actions = np.vstack((self.last_actions, ACTIONS.index(action)))
            #print(action,1111)
            return action
        else:
            if os.path.isfile("states.csv"):
                self.logger.debug("Querying model for action.")
                action = np.argmax(self.regression.predict(np.asarray(state).reshape(1,-1)))
                self.last_actions = np.vstack((self.last_actions, action))
                #print(action,"aaaa")
                return ACTIONS[action]
            else:
                self.logger.debug("Choosing action purely at random.")
                action = np.random.choice(ACTIONS, p=[.25, .25, .25, .25])
                self.last_actions = np.vstack((self.last_actions, ACTIONS.index(action)))
                #print(action,1111)
                return action
    else:
        self.logger.debug("Querying model for action.")
        #print(self.model)
        action = np.argmax(self.regression.predict(np.asarray(state).reshape(1,-1)))
        #sprint(action,"aaaa")
        return ACTIONS[action]
        

