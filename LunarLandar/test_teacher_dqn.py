#this just made sure my teacher and student were interacting correctly.. not necessary 

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from teacher_student_comparison import comparing_teacher_student
env = gym.make('LunarLander-v2')
env.seed(0)


from dqn_agent import Agent

student_agent = Agent(state_size=8, action_size=4, seed=0)

teacher_agent = Agent(state_size=8, action_size=4, seed=0)
teacher_agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))


for i in range(5):
    state = env.reset()
    ret = 0
    for j in range(1000):
        action = teacher_agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        ret += reward
        if done:
            print(ret)
            break 
env.close()
