'''
This is code for a fixed agent. The agent will be given N fixed trajectories. 
N corresponding to the number of epsidoes.
There is no TAMER learning update occuring.
The goal here is to determine where the human user is giving feedback and when.
Where corresponds to the specific state.
When corresponds to the time step. 
'''
from lavaworld_environment import grid
from tameragent import TamerAgent

def start(env_name):
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
    global myagent 
    myagent= TamerAgent()   
    if env_name == 'Lava World':
        env = grid()
    
    return env

def step(env, reward:str, episode_number):
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

    action_index = env.act(episode_number, myagent.time_step) #this will output the integer value corresponding to the action
    action_movement = env.action_list[action_index][0]
    
    myagent.current_action = [action_movement, action_index]
   

    if reward != 'None':
        updated = True
        myagent.num_feedback+=1
    else:
        updated = False

    next_state, _, done = myagent.step(myagent.current_state, action_movement, env)
    myagent.time_step+=1

    # and updated

    envState = {'done': 0, 'state': myagent.current_state, 'action': action_index, 'reward': reward, 'updated': updated, 'step': myagent.time_step, 'total_feedback_provided': 0}
    #this will send back the state feedback was provided at, when feedback was provided in terms of time steps, and the actual feedback
    
    if myagent.time_step >= 200 or done:
        envState['done']= True
        envState['total_feedback_provided'] = myagent.num_feedback
        return envState
    else:
        envState['done']= False
    

    myagent.current_state = next_state #updating the current state and action with the state we just moved to and the action we will use now


    return envState

def render_state():
    '''
       
        Caller:
        - Trial.get_render()
        Inputs:
        -  nothing
        Returns:
        - return the state tuple of the grid
     
        '''
    return tuple(myagent.current_state) #return the tuple here

def render_action():
    '''
       
        Caller:
        - Trial.get_render()
        Inputs:
        -  nothing
        Returns:
        - return the state tuple of the grid
     
        '''
    return myagent.current_action[1] #returns an integer correpsonding to the action

def reset():
    '''
        Resets the environment to start new episode.
        Caller:
        - Trial.reset()
        Inputs:
        - nothing
        Returns:
        No Return
        '''
    myagent.time_step=0
    myagent.current_state = myagent.start_state


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





