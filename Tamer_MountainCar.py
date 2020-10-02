#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval                        
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" +                " size:" + str(self.size) +                " overfullCount:" + str(self.overfullCount) +                " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)
    
    def fullp (self):
        return len(self.dictionary) >= self.size
    
    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

from math import floor, log
from itertools import zip_longest

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


# In[ ]:





# In[ ]:


import numpy as np
import itertools
import matplotlib.pyplot as plt
import gym


# In[ ]:





# In[ ]:


class MountainCarTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
       
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same

        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        """
    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.
        
        returns:
        tiles - np.array, active tiles
        """
        # Use the ranges above and self.num_tiles to scale position and velocity to the range [0, 1]
        # then multiply that range with self.num_tiles so it scales from [0, num_tiles]
        minP=-1.2
        maxP=.5
        minV=-.07
        maxV=.07
        scaleP= maxP- minP
        scaleV= maxV-minV
        
        position_scaled = ((position-minP)/(scaleP))*self.num_tiles
        
        velocity_scaled = ((velocity-minV)/(scaleV))*self.num_tiles
       
        
        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        mytiles = tiles(self.iht, self.num_tilings, [position_scaled, velocity_scaled])
        
        return np.array(mytiles)


# In[ ]:


class TamerAgent:
    """
    Initialization of Tamer Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """
    def __init__(self):
        self.last_action = None
        self.last_state = None
            
        self.previous_tiles = None
    
        self.num_tilings =  8
        self.num_tiles =  8
        self.iht_size =  4096
        self.epsilon = 0.1
        self.gamma = 1.0 # this is discount
        self.x = .32
        self.alpha =self.x/self.num_tilings  #this is step size
        self.initial_weights =  0.0
        self.num_actions = 3
        self.actions = list(range(self.num_actions))
        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
        
        # We initialize self.mctc to the mountaincar verions of the 
        # tile coder that we created
        self.mctc = MountainCarTileCoder(iht_size=self.iht_size, 
                                         num_tilings=self.num_tilings, 
                                         num_tiles=self.num_tiles)

        
    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return np.random.choice(ties)   
        
    def select_action(self, tiles):
        """
        Selects an action using epsilon greedy
        Args:
        tiles - np.array, an array of active tiles
        Returns:
        (chosen_action, action_value) - (int, float), tuple of the chosen action
                                        and it's value
        """
        action_values = []
        chosen_action = None

        for a in range(self.num_actions):
            action_values.append(np.sum(self.w[a][tiles]))
        # First loop through the weights of each action and populate action_values
        # with the action value for each action and tiles instance
        
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.actions)
        else:
            #values = self.q_values[state]
            chosen_action = self.argmax(action_values)
        return chosen_action, action_values[chosen_action]
    
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state observation from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        position, velocity = state
       # print('position', position, 'velocity', velocity)

        active_tiles=self.mctc.get_tiles(position, velocity)
       # print('active tiles', active_tiles)
        #active_tiles=mctc.get_tiles(position, velocity)


        current_action, current_action_values= self.select_action(active_tiles)
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
       
        return self.last_action
        
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        # choose the action here
        position, velocity = state
        
        # Use self.tc to set active_tiles using position and velocity
        # set current_action and action_value to the epsilon greedy chosen action using
        # the select_action function above with the active tiles
        
      #  print('check')
        active_tiles=self.mctc.get_tiles(position, velocity)
      #  print('check2')
        current_action, current_action_values= self.select_action(active_tiles)
    
        
        target = reward + current_action_values
        
        self.w[self.last_action][self.previous_tiles]+= self.alpha*(target-np.sum(self.w[self.last_action][self.previous_tiles]))
        
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action
    
    def agent_end(self, reward):

        target = reward 
        
        self.w[self.last_action][self.previous_tiles]+= self.alpha*(target-np.sum(self.w[self.last_action][self.previous_tiles]))
        


# In[ ]:


import time
np.random.seed(0)

num_runs = 10
num_episodes = 1000

all_steps = []
ENV_NAME= "MountainCar-v0"
myagent =TamerAgent()

start = time.time()
env = gym.make(ENV_NAME)
rewards_per_episode=[]
all_rewards=[]
   
steps_per_episode = []
for i_episode in range(num_episodes):
    total_reward=0
    num_steps=0
    state = env.reset()
    action = myagent.agent_start(state) #returns the first action the agent takes. 
  
    
    while True:
        
        state, reward, done, info = env.step(action) #im actually moving now.
       # print('next state, reward', state, reward)
        action = myagent.agent_step(reward, state)
        total_reward+=reward

        if done:
            myagent.agent_end(reward)
            print("Episode finished after {} timesteps".format(num_steps+1))
            steps_per_episode.append(num_steps)
            rewards_per_episode.append(total_reward)           
            break
            
        num_steps+=1
    #myagent.epsilon*=.99
    all_steps.append(np.array(steps_per_episode))
    all_rewards.append(np.array(rewards_per_episode))
print("Run time: {}".format(time.time() - start))

             


# In[ ]:




