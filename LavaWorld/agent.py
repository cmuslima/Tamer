


import gym
import time
import numpy as np
import itertools
from lavaworld_environment import grid
from tameragent import TamerAgent

#HIPPOGYM

def start():
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
    np.random.seed(0)
    global myagent 
    myagent= TamerAgent()
    env = grid()
    
    return env

def step(env, reward:str):
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

    if myagent.time_step == 0:
        myagent.agent_start() 
        
    myagent.time_step+=1
    
    current_state = myagent.current_state
    current_action_index = myagent.current_action[1]
    current_action_movement = myagent.current_action[0]

    myagent.learning_update(reward, current_state, current_action_index) #this updates the state and action we just had with feedback
    
    if reward != 'None':
        updated = True
    else:
        updated = False

    next_state, _, done = myagent.step(current_state, current_action_movement, env)
    

    envState = {'done': 0,' reward': reward, 'updated': updated, 'step': myagent.time_step}
    
    if myagent.time_step >= 200:
        envState['done']= True
        return envState
    
    else:
        envState['done']= False

    myagent.last_action= myagent.current_action  # I'm about to take a new step, so I have to put my previous state and action into a last action/state buffer
    myagent.last_state= myagent.current_state    



    new_action_movement, new_action_index = myagent.act(next_state)

    myagent.current_state = next_state #updating the current state and action with the state we just moved to and the action we will use now
    myagent.current_action = [new_action_movement, new_action_index]



    return envState

def render():
    '''
        Gets render from gym.
        Caller:
        - Trial.get_render()
        Inputs:
        -  nothing
        Returns:
        - return the state tuple of the grid
     
        '''
    return tuple(myagent.current_state) #return the tuple here
    #return env.render('rgb_array') #return the tuple here

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





