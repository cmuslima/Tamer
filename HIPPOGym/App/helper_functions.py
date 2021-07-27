import gym
import gym_minigrid
from gym_minigrid.wrappers import *
import numpy as np


def make_env(env_key, seed=None):
    env = gym.make(env_key)
    env = ImgObsWrapper(env)
    env.seed(seed)
    return env

def change_state(raw_state):
    modified_state = np.reshape(raw_state, (147,)) 
    return modified_state

def convert_reward(reward):
    if reward == 'good':
        r = 1
    elif reward == 'bad':
        r = -1
    elif reward == None:
        r = 0

    print('converted reward', r)
    if r!=0:
        update = True
    else:
        update = False
    
    return r, update