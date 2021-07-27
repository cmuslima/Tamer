#this is the tamer agent class
import numpy as np 
import random
class TamerAgent:
    """
        Initialization of Tamer Agent. All values are set to None so they can
        be initialized in the agent_init method.
        """
    def __init__(self):
        self.discount=1.0
        self.alpha=0.5
        self.eps=0.01
        self.rows = 6
        self.columns = 9
        self.q_matrix= dict()
        self.num_actions=4

        self.up=np.array([-1,0])  #0
        self.down=np.array([1, 0]) # 1
        self.left=np.array([0, -1]) # 2
        self.right=np.array([0, 1])  #3
        self.action_list=[(self.up, 0),(self.down, 1), (self.left,2), (self.right, 3)]

        self.last_action = [None, None]
        self.last_state= None
        self.current_action = [None,None]
        self.current_state= None
        self.num_feedback = 0

    
        self.start_state= None
        self.actions = list(range(self.num_actions))
        self.time_step=0

    def initalize_q_matrix(self):
        for row_num in range(self.rows):
            for col_num in range(self.columns):
                self.q_matrix.update({(row_num,col_num): np.zeros([1,self.num_actions])})        
        
    def e_greedy_action_selection(self,state):
        state_tuple= tuple(state)
        # Epsilon-greedy action selection
        if random.random() > self.eps:
            action = np.argmax(self.q_matrix[state_tuple])
        else:
            action = random.choice(np.arange(self.num_actions))


        movement, index_of_action = self.action_list[action]

        return movement, index_of_action



    def act(self, state):
        movement, index_of_action =self.e_greedy_action_selection(state)
        return movement, index_of_action

    def learning_update(self, reward, state,index_of_action): 
        if reward == 'good':
            r = 1
        elif reward == 'bad':
            r = -1

        elif reward == 'None':
            return
        
        target = r 
        
        Q_A_value= self.q_matrix[tuple(state)][0][index_of_action] 
        
        self.q_matrix[tuple(state)][0][index_of_action] = Q_A_value + self.alpha*(target- Q_A_value)
 
    def step(self, state, action_movement, env): #should return next_state, reward, done
                
    
        next_state= action_movement + state #this applies the action and moves to the next state 
        
        next_state = env.check_state(next_state, state) #this checks whether the state is hitting a blocked state
                                            # or if the state is hitting the edge, if so the next_state
                                            # is the original state
        
        done = env.check_reward(next_state)
        

        return next_state, done

    def agent_start(self):
        index_of_current_action = np.random.randint(self.num_actions)
                
        self.current_action[0]= self.action_list[index_of_current_action][0]
        self.current_action[1]= index_of_current_action
        return self.current_action
