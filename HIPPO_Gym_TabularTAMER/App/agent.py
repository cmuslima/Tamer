'''
This is code for a fixed agent. The agent will be given N fixed trajectories. 
N corresponding to the number of epsidoes.
There is no TAMER learning update occuring.
The goal here is to determine where the human user is giving feedback and when.
Where corresponds to the specific state.
When corresponds to the time step. 

Issue: having difficuly providing feedback to the first state. 
'''
from lavaworld_env import grid
from tamerAgent import TamerAgent
import time

class Agent():

    def start(self, game:str):
        '''
            Starts an OpenAI gym environment.
            Caller:
            - Trial.start()
            Inputs:
            -   game (Type: str corresponding to allowable gym environments)
            Returs:
            - env (Type: OpenAI gym Environment as returned by gym.make())
            Mandatory
        '''
        self.myagent= TamerAgent()
        self.myagent.initalize_q_matrix()   
        if game == 'Lava_World':
            self.env = grid()
            
        else:
            self.env = None
        return 
  
    
    def step(self, reward:str, episode_number):
        ##action:int is updated to reward: string
        '''
            Takes a game step.
            Caller:
            - Trial.take_step()
            Inputs:
            - env (Type: OpenAI gym Environment)
            - reward from the user (Type: string)
            Returns:
            - envState (Type: dict containing all information to be recorded for future use)
            change contents of dict as desired, but return must be type dict.
            '''
        envState = {'done': 0, 'lava': False, 'reached_goal': False,'S': "None", 'A': "None", 'R': "None", 'updated': "None", 'total_feedback_provided': 0, 'num_steps': "None", 'env_return': 0, 'q_matrix': "None", 'episode_complete': "None"}

        if self.time_steps == 0:
            self.current_action = self.myagent.agent_start()
            reward = "None"
           
    
        if reward != "None":
            updated = True
            self.myagent.num_feedback+=1
        else:
            updated = False
            
        a = get_action_in_human_readable_form(self)
 
        self.myagent.learning_update(reward, self.myagent.last_state, self.myagent.last_action[1])
        envState['S'] = self.myagent.last_state
        envState['A'] = a
        envState['R'] = reward
        envState['updated'] = updated
        print('Feedback: ', reward, 'for S, A:', self.myagent.last_state, self.myagent.last_action, 'updated', updated, 'time step:', self.myagent.time_steps)
        

 #this is a sloppy fix to the problem of not being able to provide feedback to the agent when it goes into the lava
        if self.done:
            envState = update_envState_once_done(self, envState)
            print('end of episode')
            if self.lava:
                envState['lava'] = True
                envState['reached_goal'] = False
                return envState, -1 , "IN LAVA!"
            if self.won:
                envState['lava'] = False
                envState['reached_goal'] = True
                return envState, -1 , "YOU WON!"
          



        self.myagent.last_state = self.env.agent_state #I have to remember this b/c in the env.step function, it changes the current agent state to the new one
        self.myagent.last_action = self.myagent.current_action

        next_state, done, r = self.env.step(self.myagent.current_action[0])

        #q_learning_update here
        self.myagent.q_learning_update(r, self.myagent.last_state ,self.myagent.current_action[1], next_state)


        self.score+=r

        self.myagent.current_state = next_state
        #print('new state', self.myagent.current_state)

    
        #-- this checks the end conditions and updates envState accordingly--
        if self.time_steps >= 200 or done:
            print('in this condition')
            envState = check_end_conditions(self, envState)
            return envState, self.myagent.last_action, str(envState['reached_goal'])
        else:
            envState['done']= False
            self.won = False
            self.done = False
        
    
        self.time_steps+=1
        #movement, index_of_action = self.myagent.act(self.myagent.current_state)
        movement, index_of_action = self.myagent.weighted_action(self.myagent.current_state)
        self.myagent.current_action = [movement, index_of_action]

        
        #print('new action', self.myagent.current_action[1])

        return envState, self.myagent.last_action[1], str(envState['reached_goal'])

    def render(self):

        return self.env.render()

    def reset(self):
        print('starting a new episode')
        self.time_steps=0
        self.score = 0
        self.lava = False
        self.won = False
        self.done = False
        self.myagent.last_action = [None,None]
        self.myagent.last_state= None
        self.myagent.current_action = [None,None]
        self.myagent.current_state= None
        self.score = 0
        self.env.reset()

    def close(env): #?? not right yet
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

def check_end_conditions(self, envState):
    self.done = True
    envState['done'] = True
    if tuple(self.env.agent_state) == tuple(self.env.termination_state):
        self.won = True
        envState['reached_goal'] = True
        envState['lava'] = False
    if tuple(self.env.agent_state) in self.env.blocked_states:
        envState['lava'] = True
        self.lava = True
        self.won = False
    return envState

def update_envState_once_done(self, envState):
    envState['done']= True
    envState['total_feedback_provided'] = self.myagent.num_feedback
    envState['num_steps'] = self.time_steps
    envState['env_return'] = self.score
    envState['q_matrix'] = self.myagent.q_matrix
    envState['episode_complete'] = True
    return envState

def get_action_in_human_readable_form(self):
    a = None
    if self.myagent.last_action[1] == 0:
        a = 'Up'
    if self.myagent.last_action[1] == 1:
        a = 'Down'
    if self.myagent.last_action[1] == 2:
        a = 'Left'
    if self.myagent.last_action[1] == 3:
        a = 'Right'
    if self.myagent.last_action[1] == None:
        a = 'None'
    return a
