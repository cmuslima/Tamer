import random
import numpy as np
from fixed_traj import trajectories
import gym

COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

class grid(gym.Env):
    def __init__(self):       
        self.start_state= np.array([0,0])  #target start state       
        self.termination_state= np.array([5,8]) #list that has the two termination locations
        self.blocked_states= [(1,2), (2,2),(3,2),(4,5), (0,7),(1,7), (2,7)]
        self.agent_state = self.start_state
        self.rows=6
        self.columns=9
        #actions 
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.action_sequence = [(1,1,1,1,1,3,3,3,3,3,3,3,3)]
        self.tile_size = 64

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

    def step(self, action_movement): #should return next_state, reward, done
                
    
        next_state= action_movement + self.agent_state #this applies the action and moves to the next state 
        
        next_state = self.check_state(next_state, self.agent_state) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
        
        self.agent_state = next_state
        done = self.check_reward(next_state)
        

        return next_state, done

    def fill_square(self, row, col, color, img):
        img[row*self.tile_size:(row+1)*self.tile_size,col*self.tile_size:(col+1)*self.tile_size] = color
        return img
        
    def render(self):
        width_px = self.columns * self.tile_size
        height_px = self.rows * self.tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        # Color background grey
        img[:,:,:] = COLORS['grey']
        # Color lava red
        for lava in self.blocked_states:
            img = self.fill_square(lava[0], lava[1], COLORS['red'], img) 
        # Color agent blue
        img = self.fill_square(self.agent_state[0],self.agent_state[1], COLORS['blue'],img)
        # Color goal state green
        img = self.fill_square(self.termination_state[0],self.termination_state[1],COLORS['green'],img)

        return img



a = 12

myenv = grid()
import matplotlib.pyplot as plt
plt.figure()
#plt.imshow(np.zeros((3,3)))
plt.imshow(myenv.render())
plt.show()