import random
import numpy as np
from uniformadvising import uniform_advising
import pickle

def main(budget, uniform_value, credit_assignment_value):
    print('We are running an experiment on Mountain Car')
    print('Our teacher agent has a budget of:', budget)
    print('The teacher will provide feedback every ', uniform_value, ' time steps')
    print('The credit assignment value is:', credit_assignment_value)
    
    num_runs = 1
    num_episodes = 100

    run_steps =[]
    eval_score_per_run =[]

    for i in range(1, num_runs+1):
        print('Experiment #', i)
        scores,evaluation_scores = uniform_advising.main_loop(uniform_value, budget, num_episodes, credit_assignment_value)
        run_steps.append(scores)
        eval_score_per_run.append(np.array(evaluation_scores))

    #This is the data we are saving.
    #This is a list with 3 componenets 
    #1. The credit assignment value
    #2. The average performance of the student agent
    #3. The standard deviation of the performance 
    average_evaluation_scores = [ 'creditvalue '+str(credit_assignment_value),np.mean(eval_score_per_run, axis=0), np.std(eval_score_per_run, axis=0)]


    #This is how we save the data to a file 
    model_name = 'average_eval_scores' + 'b_'+ str(budget) + 'uniform_'+str(uniform_value) + 'credit_assignment_value' + str(credit_assignment_value) + '.pkl'
    with open(model_name,'wb') as output:
        pickle.dump(average_evaluation_scores, output)
    



    return average_evaluation_scores

