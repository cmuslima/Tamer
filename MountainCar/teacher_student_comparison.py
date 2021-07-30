import mctc
import numpy as np
class comparing_teacher_student:

    def uniform_update(teacher_agent, student_agent, student_action, state, uniform_value, budget_used, max_budget, total_time_step):
        teacher_active_tiles = comparing_teacher_student.return_tiles(teacher_agent, state)
        if total_time_step % uniform_value == 0:
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, student_action, budget_used, max_budget,teacher_active_tiles)
            
            if reward!= None:
                #Credit assignment version:
                student_agent.learn(reward)

                #No credit assignment version
                #student_agent.w[student_agent.last_action][student_agent.previous_tiles]+= (student_agent.alpha)*(reward-student_agent.w[student_agent.last_action][student_agent.previous_tiles])

        else:
            reward = None
            update = 0 

        return reward, update

    def get_importance_value(teacher_agent, tiles):
        action_values = []
        num_actions = 3
        for a in range(num_actions):
            action_values.append(np.sum(teacher_agent.w[a][tiles]))
        
        iv = abs(max(action_values)-min(action_values))
        return iv

    def return_tiles(teacher_agent, state):
        position, velocity = state 
        active_tiles = teacher_agent.mctc.get_tiles(position, velocity)
        return active_tiles


    def teacher_best_action(teacher_agent, active_tiles):

        action, _ = teacher_agent.select_greedy_action(active_tiles)

        return action


    def compare_actions_budget(teacher_agent, student_action, budget_used, max_budget, active_tiles):
        teacher_action = comparing_teacher_student.teacher_best_action(teacher_agent, active_tiles)
        if budget_used <= max_budget:
            if teacher_action == student_action:
                reward = 1
            else: 
                reward = -1
            update = 1
        else:
            reward = None
            update = 0

        return reward, update 