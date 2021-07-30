import numpy as np
from TileCodingAPI import IHT
from mctc import MountainCarTileCoder

class SarsaAgent:
    """
    Initialization of Sarsa Agent. All values are set to None so they can
    be initialized in the agent_init method.
    """
    def __init__(self):
        self.last_action = None
        self.last_state = None
        self.test = 5
        self.previous_tiles = None
    
        self.num_tilings =  8
        self.num_tiles =  8
        self.iht_size =  4096
        self.epsilon = 0.1
        self.gamma = 1.0 # this is discount
        self.x = .2
        self.alpha = self.x/self.num_tilings  #this is step size
       
        self.initial_weights =  0.0
        self.num_actions = 3
        self.actions = list(range(self.num_actions))
        self.visit_count=dict()
        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.ones((self.num_actions, self.iht_size)) * self.initial_weights
        
        # We initialize self.mctc to the mountaincar verions of the 
        # tile coder that we created
        self.mctc = MountainCarTileCoder(iht_size=self.iht_size, 
                                         num_tilings=self.num_tilings, 
                                         num_tiles=self.num_tiles)
        self.importance_values=[]

        
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
        
        
        #this code updates the visit count of the active tiles
        if str(tiles) in self.visit_count.keys():    
            self.visit_count[str(tiles)]=self.visit_count.get(str(tiles))+1
        else:
            self.visit_count[str(tiles)]=1
        
        self.importance_values.append(abs(max(action_values)-min(action_values)))
        if np.random.random() < self.epsilon:
            chosen_action = np.random.choice(self.actions)
        else:
            #values = self.q_values[state]
            chosen_action = self.argmax(action_values)
        return chosen_action, action_values[chosen_action]

    def select_greedy_action(self, tiles):
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

        active_tiles=self.mctc.get_tiles(position, velocity)
      #  print('check2')
        current_action, current_action_values= self.select_action(active_tiles)
    
        
        target = reward + current_action_values #i just made this change
        
        self.w[self.last_action][self.previous_tiles]+= .01*(target-np.sum(self.w[self.last_action][self.previous_tiles]))
        
        
        self.last_action = current_action
        self.previous_tiles = np.copy(active_tiles)
        return self.last_action
    
    def agent_end(self, reward):

        target = reward 
        
        self.w[self.last_action][self.previous_tiles]+= self.alpha*(target-np.sum(self.w[self.last_action][self.previous_tiles]))

    def evaluation_agent_step(myagent, state):
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

        active_tiles=myagent.mctc.get_tiles(position, velocity)

        current_action, current_action_values= myagent.select_action(active_tiles)


        #target = reward + current_action_values #i just made this change

        #self.w[self.last_action][self.previous_tiles]+= .01*(target-np.sum(self.w[self.last_action][self.previous_tiles]))


        myagent.last_action = current_action
        myagent.previous_tiles = np.copy(active_tiles)
        return myagent.last_action