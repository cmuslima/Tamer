import random
import numpy as np
from fixed_traj import trajectories
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
        self.action_sequence = [(1,1,1,1,1,3,3,3,3,3,3,3,3)]

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
    
    def act(self, episode_number, time_step):
        return self.action_sequence[episode_number][time_step]

    def step(self, state, action_movement, env): #should return next_state, reward, done
                
    
        next_state= action_movement + state #this applies the action and moves to the next state 
        
        next_state = env.check_state(next_state, state) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
        
        done = env.check_reward(next_state)
        

        return next_state, done