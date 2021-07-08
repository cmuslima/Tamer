#Same as the run.py 
#this just runs the code with multiple cores 
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


num_cores = 5

import time
import multiprocessing


def tamer_loop(b, threshold):
    print('budget', b, 'threshold', threshold)
    num_runs = 15
    run_steps =[]
    eval_score_per_run =[]
    average_eval_score = []
    budget_ends_per_run=[]
    budget_used_per_run=[]
    env = gym.make('LunarLander-v2')
    env.seed(0)
    for i in range(num_runs):
        print('threshold', threshold)
        
        scores, evaluation_scores, budget_ends, budget_used = teacher_student_loop(n_episodes=1000, max_t=1000, eps_start=.01, eps_end=0.001, eps_decay=0.995, env=env, threshold = threshold, b=b)
        run_steps.append(np.array(scores))
        eval_score_per_run.append(np.array(evaluation_scores))
        budget_ends_per_run.append(budget_ends)
        budget_used_per_run.append(budget_used)

    average_eval_score.append((np.mean(eval_score_per_run, axis=0), np.std(eval_score_per_run, axis=0), 'threshold'+ str(threshold), 'budget:'+ str(b)))

    model_name = 'average_eval_scores' + 'b_'+ str(b) + 'uniform__'+str(threshold) + '.pkl'
    with open(model_name,'wb') as output:
        pickle.dump(average_eval_score, output)
    
    model_name = 'raw_eval_scores' + 'b_'+ str(b) + 'uniform__'+str(threshold) + '.pkl'
    with open(model_name,'wb') as output:
        pickle.dump(eval_score_per_run, output)

    model_name = 'budget_ends_per_run' + 'b_'+ str(b) + 'uniform_'+str(threshold) + '.pkl'
    with open(model_name,'wb') as output:
        pickle.dump(budget_ends_per_run, output)

    model_name = 'budget_used_per_run' + 'b_'+ str(b) + 'uniform_'+str(threshold) + '.pkl'
    with open(model_name,'wb') as output:
        pickle.dump(budget_used_per_run, output)

    return 'is done'


def multiprocessing_func_small_budget(x):
    print('{} is {} '.format(x, tamer_loop(5000, x)))

def multiprocessing_func_large_budget(x):
    print('{} is {} '.format(x, tamer_loop(10000, x)))
    
if __name__ == '__main__':
    starttime = time.time()
    pool = multiprocessing.Pool()
    x = [1,5,10]
    pool.map(multiprocessing_func_small_budget, x)
    pool.close()
    print()

    pool = multiprocessing.Pool()
    pool.map(multiprocessing_func_large_budget, x)
    pool.close()
    print()
    print('Time taken = {} seconds'.format(time.time() - starttime))


