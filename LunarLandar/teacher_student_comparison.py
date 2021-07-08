class comparing_teacher_student:
    
    def importance_update(teacher_agent, student_action, state, threshold, budget_used, max_budget, unlimited, time_of_feedback, time_step):
        action_values = teacher_agent.evaluate_q(state)
        iv=abs(max(action_values)-min(action_values))

        if iv >= threshold:
            time_of_feedback.append(time_step)
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, state, student_action, budget_used, max_budget, unlimited)
        else:
            reward = None
            update = 0 

        return reward, update
    
    def uniform_update(teacher_agent, student_action, state, uniform_value, budget_used, max_budget, unlimited, total_time_step):
        if total_time_step % uniform_value == 0:
            reward, update = comparing_teacher_student.compare_actions_budget(teacher_agent, state, student_action, budget_used, max_budget, unlimited)
        else:
            reward = None
            update = 0 

        return reward, update

    def teacher_best_action(teacher_agent, state):
        eps = 0
        action = teacher_agent.act(state, eps)

        return action

    def compare_actions(teacher_agent, state, student_action):

        teacher_action = comparing_teacher_student.teacher_best_action(teacher_agent, state)
        
        if teacher_action == student_action:
            reward = 1
        else: 
            reward = -1
        
        return reward

    def compare_actions_budget(teacher_agent, state, student_action, budget_used, max_budget, unlimited):
        teacher_action = comparing_teacher_student.teacher_best_action(teacher_agent, state)
        
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
