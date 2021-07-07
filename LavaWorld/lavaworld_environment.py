#Basic set up for the Lava World environment. 

import random
import numpy as np
class grid():
    def __init__(self):       
        self.start_state= np.array([0,0])  #target start state       
        self.termination_state= np.array([5,8]) #list that has the two termination locations
        self.blocked_states= [(1,2), (2,2),(3,2),(4,5), (0,7),(1,7), (2,7)]
        self.rows=6
        self.columns=9
        #actions 
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3

        
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]


    def check_reward(self, state):
        terminal= False
        
        if np.array_equal(state,self.termination_state) == True:
            terminal= True
        
        return terminal 

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
        
        if next_state_tuple in self.blocked_states: # bc if this happens it wouldnt be near a blocked state 
            next_state = state   
            
        elif next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state 
            
        return next_state
