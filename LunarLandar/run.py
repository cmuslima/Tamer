#currently, this run.py is using uniform advising. 

import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from teacher_student_comparison import comparing_teacher_student
from dqn_agent import DQNAgent
from teacher_student_interaction import teacher_student_loop
import pickle


env = gym.make('LunarLander-v2')
env.seed(0)
num_runs = 1
run_steps =[]
threshold_parameters = [25]
budgets = [1000]
eval_score_per_run = []
average_eval_score = []
for b in budgets:
    for threshold in threshold_parameters:
        for i in range(num_runs):
            print('threshold', threshold)
            scores, evaluation_scores = teacher_student_loop(n_episodes=500, max_t=1000, eps_start=.01, eps_end=0.001, eps_decay=0.995, env=env, threshold = threshold, b=b)
            run_steps.append(np.array(scores))
            eval_score_per_run.append(np.array(evaluation_scores))
            print('len of evaluation scores', evaluation_scores)
            print()
        average_eval_score.append((np.mean(eval_score_per_run, axis=0), np.std(eval_score_per_run, axis=0), 'threshold'+ str(threshold), 'budget:'+ str(b)))

        model_name = 'average_eval_scores' + 'b_'+ str(b) + 'uniform_'+str(threshold) + '.pkl'
        with open(model_name,'wb') as output:
            pickle.dump(average_run_score, output)
    
# plot the scores
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(np.arange(len(average_run_score[0][0])), average_run_score[0][0])

#plt.ylabel('Score')
#plt.xlabel('Episode #')
#plt.savefig('iv.png')
