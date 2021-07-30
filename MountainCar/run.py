'''Instead of using an actual human teacher, we can use the artifical teacher.
Deep TAMER paper says that they observed humans provide feedback every 25 time steps. So our
artifical teacher can 'mimic' the human teacher by providing feedback at a similar pace.
So we can set our uniform value to 25. This means the artifical teacher will provide feedback every
25 time steps. Then we can vary the credit assignment parameter and we can see how this affects performance.
Does assigning feedback to a large list of previous states/actions help or hurt the student agent?'''


'''Cite Deep TAMER: Interactive Agent Shaping in High-Dimensional State Spaces'''
import sys
import argparse
from main import main
import matplotlib.pyplot as plt
import numpy as np

# Argument definition
parser = argparse.ArgumentParser()

parser.add_argument('--uniformvalue', '-u', type=int, default=20)
parser.add_argument('--creditassignmentvalue', '-ca', type=int, default=5)
parser.add_argument('--budget','-b', type=int, default=100)
args = parser.parse_args()

average_evaluation_scores = main(args.budget, args.uniformvalue, args.creditassignmentvalue)
#recall that we stored the mean scores across 30 experiments and we stored the standard deviation
#average_evaluation_scores[0] just gives us the mean values



#This plots the student performance when using the specific credit assignment value
fig = plt.figure()
length = len(average_evaluation_scores[1])
group = average_evaluation_scores[1] 
label = 'credit value=' + str(args.creditassignmentvalue)
print('label', label)
plt.plot(np.arange(length), group, lw = 2, color = 'blue', label = label)
plt.legend() 
plt.ylim(0,220)
plt.xlabel('episode #')
plt.ylabel('score')
plt.title('Performance in Mountain Car with Credit Assignment Value of ' + str(args.creditassignmentvalue))
plt.savefig('MountainCarPerformance_CA_'+str(args.creditassignmentvalue)+ '.png')