import mctc
import numpy as np
class comparing_teacher_student:
    def uniform_update(teacher_agent, student_agent, student_action, state, uniform_value, budget_used, max_budget, unlimited, total_time_step):
        teacher_active_tiles = comparing_teacher_student.return_tiles(teacher_agent, state)
        if total_time_step % uniform_value == 0:
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, student_action, budget_used, max_budget, unlimited, teacher_active_tiles)
            
            if reward!= None:
                student_agent.w[student_agent.last_action][student_agent.previous_tiles]+= (student_agent.alpha)*(reward-student_agent.w[student_agent.last_action][student_agent.previous_tiles])

        else:
            reward = None
            update = 0 

        return reward, update
    def update_student_dict(student_agent, tiles,max_updates):
        #this tells us which active tiles were given feedback 
        if str(tiles) in student_agent.visit_count.keys():    
            student_agent.visit_count[str(tiles)]=student_agent.visit_count.get(str(tiles))+1
        else:
            student_agent.visit_count[str(tiles)]=1
        
        num = student_agent.visit_count.get(str(tiles))
        #print('num', num)
        if num > max_updates:
            reached_feedback_limit = True
            #print('reached limit')
        else:
            reached_feedback_limit = False
        return reached_feedback_limit

    def count_update(teacher_agent, student_agent, student_action, state, threshold, budget_used, max_budget, unlimited,max_updates):
        teacher_active_tiles = comparing_teacher_student.return_tiles(teacher_agent, state)

        if str(teacher_active_tiles) not in teacher_agent.visit_count.keys():
            reward = None
            update = 0
            return reward, update

        if teacher_agent.visit_count[str(teacher_active_tiles)]> threshold:
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, student_action, budget_used, max_budget, unlimited, teacher_active_tiles)    
            reached_feedback_limit = comparing_teacher_student.update_student_dict(student_agent, teacher_active_tiles, max_updates)

            if reward == None or reached_feedback_limit == True:
                update = 0
                #print('can not update b/c I reached the limit')
                #print('update', update)
                return reward, update
            

            student_agent.w[student_agent.last_action][student_agent.previous_tiles]+= (student_agent.alpha)*(reward-student_agent.w[student_agent.last_action][student_agent.previous_tiles])
        else:
            reward = None
            update = 0 

        return reward, update

    def iv_update(teacher_agent, student_agent, student_action, state, threshold, budget_used, max_budget, unlimited,max_updates):
        teacher_active_tiles = comparing_teacher_student.return_tiles(teacher_agent, state)
        iv = comparing_teacher_student.get_importance_value(teacher_agent, teacher_active_tiles)

        if iv > threshold:
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, student_action, budget_used, max_budget, unlimited, teacher_active_tiles)    
            reached_feedback_limit = comparing_teacher_student.update_student_dict(student_agent, teacher_active_tiles, max_updates)

            if reward == None or reached_feedback_limit == True:
                update = 0
                #print('can not update b/c I reached the limit')
                #print('update', update)
                return reward, update
            
            #this tells us which active tiles were given feedback 

            student_agent.w[student_agent.last_action][student_agent.previous_tiles]+= (student_agent.alpha)*(reward-student_agent.w[student_agent.last_action][student_agent.previous_tiles])
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

    def compare_actions(teacher_agent, state, student_action):

        teacher_action = comparing_teacher_student.teacher_best_action(teacher_agent, state)

        if teacher_action == student_action:
            reward = 1
        else: 
            reward = -1
        
        return reward

    def compare_actions_budget(teacher_agent, student_action, budget_used, max_budget, unlimited, active_tiles):
        teacher_action = comparing_teacher_student.teacher_best_action(teacher_agent, active_tiles)
        if unlimited == False:
            if budget_used <= max_budget:
                if teacher_action == student_action:
                    reward = 1
                else: 
                    reward = -1
                update = 1
            else:
                reward = None
                update = 0
        else:
            if teacher_action == student_action:
                reward = 1
            else: 
                reward = -1
            
            update = 1

        return reward, update 