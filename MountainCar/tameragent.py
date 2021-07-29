import numpy as np
from TileCodingAPI import IHT
from mctc import MountainCarTileCoder

class TamerAgent:
    """
        Initialization of Tamer Agent. All values are set to None so they can
        be initialized in the agent_init method.
        """
    def __init__(self):
        
        self.last_action = None
        self.previous_tiles = None
        #self.first_state= None
        self.current_action = None
        self.current_tiles= None
        self.state= None 
        self.num_tilings =  8
        self.alpha=1/16 #1/self.num_tilings
        
        self.num_tilings =  16
        self.num_tiles =  8
        self.iht_size =  4096
        self.epsilon = 0
        self.num_actions = 3
        self.actions = list(range(self.num_actions))
        self.time_step=0
        self.IV=None
        self.visit_count =dict()
        self.experiences= list()
        self.max_n_experiences=5
        self.step_size = 1/self.num_tilings
        
        # We initialize self.w to three times the iht_size. Recall this is because
        # we need to have one set of weights for each action.
        self.w = np.zeros((self.num_actions, self.iht_size))
        
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

    def select_greedy_action(self, tiles):
        """
            Selects an action using greedy
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
            chosen_action = self.argmax(action_values)
        
        return chosen_action
    
    def action_selection(self, state):
        position, velocity = state
        active_tiles=self.mctc.get_tiles(position, velocity)
        current_action=self.select_greedy_action(active_tiles)
        #print(active_tiles)
        self.current_action = current_action
        self.current_tiles = np.copy(active_tiles)
        self.update_experiences(active_tiles, current_action)
    
    
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
            the environment starts.
            Args:
            state (Numpy array): the state observation from the
            environment's evn_start function.
            Returns:
            The first action the agent takes.
            """

        self.state=state
        position, velocity = state
        
        active_tiles=self.mctc.get_tiles(position, velocity)
        
        self.current_action = np.random.choice(self.actions)
        self.current_tiles= np.copy(active_tiles)

        self.update_experiences(self.current_tiles,self.current_action)
        
        return self.current_action

    def update_experiences(self, tiles, action):

        if len(self.experiences) == self.max_n_experiences:
            self.experiences.pop(0)#this makes sure we are always keeping tracking of the last n experiences.. we remove the oldest one from the experiences buffer
        
        self.experiences.append((tiles, action)) # now we add in the newest one to the buffer

    

    def update_reward(self, artificial_agent):
        position, velocity = self.state
        active_tiles_artifical = artificial_agent.mctc.get_tiles(position, velocity)
    
        artifical_action, artifical_action_values= artificial_agent.select_action(active_tiles_artifical)

        if artifical_action==self.last_action:
            target = 1
            #print('right action')
        else:
            target = -1
            #print('wrong action')
 
#below is credit assignment with time
        '''current_time = time.time()
        while len(self.experiences) > 0:
            experience = self.experiences[0]
            
            #diff= current_time-experience[2]

            #if (diff < .2 or diff > 2):
            
            if experience[2] < current_time - self.window_size: #
                self.experiences.pop(0)
            
            else:
                break'''


    # update weights using Algorithm 1 in paper
        n_experiences = len(self.experiences)

        if n_experiences== 0:
            return
        weight_per_experience = 1.0/n_experiences

        cred_features = np.zeros((self.num_actions, self.iht_size))
            
  
        for experience in self.experiences:
            
            exp_features= np.zeros((self.num_actions, self.iht_size))
          
            action = experience[1]
            tile = experience[0]
            exp_features[action][tile]=1

            exp_features*=weight_per_experience

            cred_features = np.add(cred_features, exp_features)
       
        error = target - self.w * cred_features
        self.w += (self.step_size*error*cred_features)

    def render(env):
        '''
            Gets render from gym.
            Caller:
            - Trial.get_render()
            Inputs:
            - env (Type: OpenAI gym Environment)
            Returns:
            - return from env.render('rgb_array') (Type: npArray)
            must return the unchanged rgb_array
            '''
        return env.render('rgb_array')

    def reset(self, env):
        '''
            Resets the environment to start new episode.
            Caller:
            - Trial.reset()
            Inputs:
            - env (Type: OpenAI gym Environment)
            Returns:
            No Return
            '''
        self.time_step=0
        self.budget_per_episode=0 #this is reset 
        self.state = env.reset()

    def close(self, env):
        '''
            Closes the environment at the end of the trial.
            Caller:
            - Trial.close()
            Inputs:
            - env (Type: OpenAI gym Environment)
            Returns:
            No Return
            '''
        env.close()
