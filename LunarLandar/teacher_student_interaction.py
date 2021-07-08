import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from teacher_student_comparison import comparing_teacher_student

from dqn_agent import DQNAgent
from deeptameragent import TamerAgent


def initialize():
    student_agent = TamerAgent(state_size=8, action_size=4, seed=0)
    teacher_agent = DQNAgent(state_size=8, action_size=4, seed=0)
    teacher_agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    return student_agent, teacher_agent


def teacher_student_loop(n_episodes, max_t, eps_start, eps_end, eps_decay,env, threshold, b):
    """
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    student_agent, teacher_agent = initialize()
    budget_used = 0
    total_time_step = 0
    evaluation_scores =[]
    budget_ends = None
    for i_episode in range(1, n_episodes+1):
        
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = student_agent.act(state, eps)
            next_state, env_reward, done, _ = env.step(action)
            teacher_feedback, update = comparing_teacher_student.uniform_update(teacher_agent, action, state, threshold, budget_used, b, False, total_time_step)
        
            if teacher_feedback is not None:
                feedbackgiven= True
                student_agent.add_to_memory(state, action, teacher_feedback, next_state, done)
                student_agent.step(feedbackgiven) #this says I will make an update if I get a feedback.
            else: 
                feedbackgiven = False
                student_agent.step(feedbackgiven) 
                #this says I will make an update every 10 time steps. 


            budget_used+=update
            state = next_state
            score += env_reward
            total_time_step+=1
                
            if budget_used == b-1:
                budget_ends = i_episode
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        if i_episode % 10 == 0: #evaluate for 20 episodes 
            eval_scores_list = []
            for i in range(0, 20):
                state = env.reset()
                eval_score = 0
                for t in range(max_t):
                    eps = 0 
                    action = student_agent.act(state, eps)
                    next_state, env_reward, done, _ = env.step(action)
                    state = next_state
                    eval_score += env_reward
                    if done:
                        break 
                eval_scores_list.append(eval_score)
            mean_score = sum(eval_scores_list)/len(eval_scores_list)
            evaluation_scores.append(mean_score)
    if budget_ends == None:
        budget_ends = n_episodes
    model_name = 'b_'+ str(b) + 'uniform_'+str(threshold) + '.pth'
    torch.save(student_agent.qnetwork_local.state_dict(), model_name) #need to figure out a way to dynamically change this..
    return scores, evaluation_scores, budget_ends, budget_used