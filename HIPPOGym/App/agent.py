'''
This is a demo file to be replaced by the researcher as required.
This file is imported by trial.py and trial.py will call:
start()
step()
render()
reset()
close()
These functions are mandatory. This file contains minimum working versions 
of these functions, adapt as required for individual research goals.
'''
import gym
import gym_minigrid
from helper_functions import make_env, convert_reward, change_state
import random
import numpy as np
from deeptameragent import TamerAgent


"""
This is a generic agent class which is called by the the HIPPO Gym backend, in trial.py.
This agent class instantiates the TamerAgent class. The TamerAgent contains the learning update code, etc """

class Agent():
    '''
    Use this class as a convenient place to store agent state.
    '''
    
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
        seed = np.random.seed(0)
        self.eps = 1
        self.env = make_env(game, seed)

        self.tameragent= TamerAgent(state_size=147, action_size=6, seed=0)
        return self.env

    def update_agent(self, reward):
    
        reward, update = convert_reward(reward)

        self.tameragent.add_to_memory(self.tameragent.state, self.tameragent.action, reward, self.tameragent.next_state, self.tameragent.done)
        self.tameragent.step(reward) #this says I will make an update if I get a feedback.

        if update:
            self.numfeedbacks+=1
        if self.tameragent.env_reward!=0:
            self.numwins+=1

        envState = {'done': self.tameragent.done,' reward': reward} 

        if self.tameragent.done:
            envState = {'done': self.tameragent.done,' reward': reward, ' numfeedbacks': self.numfeedbacks, ' numwins': self.numwins} 

        return envState
    def step(self, reward:str):
        '''
        Trajectory looks like: S, A, R, S'
        The human will provide feedback once the agent takes an action which leads to a subsequent next state, S'.
        Because we are using Deep Tamer, we must know S' and D (Done) along with the feedback to update the experience replay buffer.
        
        So we update (S, A, S') with the feedback.
        
        Then we set S as S' and take a new action.

        Caller: 
            - Trial.take_step()
        Inputs:
            - env (Type: OpenAI gym Environment)
            - human feedback (Type: string)
        Returns:
            - envState (Type: dict containing all information to be recorded for future use)
              change contents of dict as desired, but return must be type dict.
        '''
        print('r', reward)

        if self.numtimesteps == 0:
            Agent.agent_start(self)

        envState = Agent.update_agent(self,reward) 

        oldaction = self.tameragent.action
        #i'm returning this action because I want the action to be displayed for the user
        # to see which action the tamer agent is taking before providing feedback
        #however, I think there is still a time delay, so the action displayed 
        #on the screen doesn't make directly with the time the action is actually taken
        #by the tamer agent.
        
        self.tameragent.state = self.tameragent.next_state
        
        self.eps = max(self.tameragent.eps_end, self.tameragent.eps_decay*self.eps) # decrease epsilon

        self.tameragent.action = self.tameragent.act(self.tameragent.state, self.eps)
    
        next_state, self.tameragent.env_reward, self.tameragent.done, _ = self.env.step(self.tameragent.action)

        self.tameragent.next_state = change_state(next_state)

        return envState, oldaction

    
    def render(self):
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
        return self.env.render('rgb_array')
    
    def reset(self):
        '''
        Resets the environment to start new episode.
        Caller: 
            - Trial.reset()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns: 
            No Return
        '''
        self.numwins = 0
        self.numfeedbacks = 0 
        self.numtimesteps = 0 
        self.tameragent.state = change_state(self.env.reset()) #first state
        
    
    def agent_start(self):

        self.tameragent.action = self.tameragent.act(self.tameragent.state, self.eps)
    
        next_state, self.tameragent.env_reward, self.tameragent.done, _ = self.env.step(self.tameragent.action)

        self.tameragent.next_state = change_state(next_state)

    def close(self):
        '''
        Closes the environment at the end of the trial.
        Caller:
            - Trial.close()
        Inputs:
            - env (Type: OpenAI gym Environment)
        Returns:
            No Return
        '''
        self.env.close()
