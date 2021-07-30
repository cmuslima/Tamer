import numpy as np
from tameragent import TamerAgent
from sarsaagent import SarsaAgent
import gym
import pickle
from teacher_student_comparison import comparing_teacher_student


class uniform_advising():

    def evaluation_agent_step(myagent, state):
        position, velocity = state
        active_tiles=myagent.mctc.get_tiles(position, velocity)
        current_action = myagent.select_greedy_action(active_tiles)
        myagent.last_action = current_action
        myagent.previous_tiles = np.copy(active_tiles)
        return myagent.last_action

    def initalize(credit_assignment_value):
        np.random.seed(0)
        teacher_agent=SarsaAgent()

        # arbitrary reset before loading from pickle file
        teacher_agent.mctc = 0 
        teacher_agent.w = 0

        with open('sarsa_trained_mctc.pkl','rb') as input:
            teacher_agent.mctc = pickle.load(input)
        with open('sarsa_trained_weights.pkl','rb') as input:
            teacher_agent.w = pickle.load(input)

        student_agent= TamerAgent(credit_assignment_value)
        return teacher_agent, student_agent

    def uniform_step(env, max_budget, budget_used, uniform_value, student_agent, teacher_agent, total_num_steps):
        
        if student_agent.time_step == 0:
            student_agent.agent_start(student_agent.state)
        
        student_agent.last_action= student_agent.current_action
        student_agent.previous_tiles= student_agent.current_tiles

        student_action = student_agent.last_action
        reward, update = comparing_teacher_student.uniform_update(teacher_agent, student_agent, student_action, student_agent.state, uniform_value, budget_used, max_budget, total_num_steps)
    
        student_agent.time_step+=1
        state, _ , done, _ = env.step(student_agent.current_action)
        student_agent.state=state
        student_agent.action_selection(state)
        
        return reward, update, done 

    def main_loop(uniform_value, budget, num_episodes, credit_assignment_value):
        teacher_agent, student_agent = uniform_advising.initalize(credit_assignment_value)
        total_num_steps=0
        budget_used=0
        steps_per_episode = []
        ENV_NAME= "MountainCar-v0"
        env = gym.make(ENV_NAME)
        evaluation_scores =[]
        for i_episode in range(num_episodes):
            num_steps=0
            student_agent.reset(env)
            while True:

                reward, update, done = uniform_advising.uniform_step(env, budget, budget_used, uniform_value, student_agent, teacher_agent, total_num_steps)
                budget_used +=update

                num_steps+=1
                total_num_steps+=1

                if done:
                    steps_per_episode.append(num_steps)
                    break

                
            if i_episode % 10 == 0: #evaluate for 20 episodes 
                eval_scores_list = []
                for i in range(0, 20):
                    num_steps=0
                    state = env.reset()
                    action = uniform_advising.evaluation_agent_step(student_agent, state)#returns the first action the agent takes. 
                    while True:
                        state, _, done, _ = env.step(action) #im actually moving now. 
                        action = uniform_advising.evaluation_agent_step(student_agent, state)
                        if done:
                            break
                        num_steps+=1
                    eval_scores_list.append(num_steps)
                mean_score = sum(eval_scores_list)/len(eval_scores_list)
                evaluation_scores.append(mean_score)

        
        return np.array(steps_per_episode), evaluation_scores

            