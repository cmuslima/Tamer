import random
import numpy as np
import gym

COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([255, 255, 255]),
    'black' : np.array([0, 0, 0]),
    'orange': np.array([255,165,0])
}

class grid(gym.Env):
    def __init__(self):       
        self.start_state= np.array([0,0])  #target start state       
        self.termination_state= np.array([5,8]) #list that has the two termination locations
        self.blocked_states= [(0,2), (1,2), (2,2), (4,3), (4,4), (4,5), (4,6), (4,7)]
        self.agent_state = self.start_state
        self.rows=6
        self.columns=9
        #actions 
        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]
        self.tile_size = 64

        self.episode_number = 0 

    def check_reward(self, state):
        terminal= False
        r = -1
        if np.array_equal(state,self.termination_state) == True:
            terminal= True
            r = 0
        if tuple(state) in self.blocked_states:
            terminal = True
            r = -100
        return terminal, r

    def check_state(self, next_state, state):
        
        next_state_tuple= tuple(next_state)
                    
        if next_state_tuple[0] == self.rows or next_state_tuple[1] == self.columns  or -1 in next_state_tuple:
            next_state = state 
            
        return next_state
    
    def act(self,time_step):

        return self.action_sequence[self.episode_number][time_step]

    def step(self, action_movement): #should return next_state, reward, done
        
        next_state= action_movement + self.agent_state #this applies the action and moves to the next state 
        
        next_state = self.check_state(next_state, self.agent_state) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
        
        self.agent_state = next_state
        done, r = self.check_reward(next_state)
        

        return next_state, done, r

    def fill_square(self, row, col, color, img):
        img[row*self.tile_size:(row+1)*self.tile_size,col*self.tile_size:(col+1)*self.tile_size] = color
        return img
        
    def l2_dist(self, point1, point2):
        xdist = point1[1]-point2[1]
        ydist = point1[0]-point2[0]
        return np.sqrt(xdist**2 + ydist**2)

    def fill_circle(self, row, col, color, img):
        center = (row*self.tile_size+self.tile_size//2, col*self.tile_size+self.tile_size//2)
        rad = self.tile_size//3
        for r in range(row*self.tile_size, (row+1)*self.tile_size):
            for c in range(col*self.tile_size, (col+1)*self.tile_size):
                if self.l2_dist((r,c),center) < rad:
                    img[r,c] = color
        return img


    def render(self):
        width_px = self.columns * self.tile_size
        height_px = self.rows * self.tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        # Color background grey
        img[:,:,:] = COLORS['white']




        # Color lava red
        for lava in self.blocked_states:
            img = self.fill_square(lava[0], lava[1], COLORS['orange'], img) 

        # Color goal state green
        img = self.fill_square(self.termination_state[0],self.termination_state[1],COLORS['green'],img)
        # Draw lines in grid
        row_ticks = list(range(1,self.rows))
        col_ticks = list(range(1,self.columns))
        for r in row_ticks:
            img[r*self.tile_size,:] = COLORS['black']
        for c in col_ticks:
            img[:,c*self.tile_size] = COLORS['black']
        
        
        # Color agent blue
        img = self.fill_circle(self.agent_state[0],self.agent_state[1], COLORS['blue'],img)
        return img

    def reset(self):
        self.agent_state = self.start_state

    