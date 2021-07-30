import matplotlib.pyplot as plt
import pickle
import numpy as np

#this function takes in a file name and returns the data that is stored within the file
#file here is the name of the file as a string.
def get_data(file): 
    with open(file,'rb') as input:
        data = pickle.load(input)      
    return data

#Every time you run an experiment, you will save data.
#Refer back to the main.py file to figure out what data is being saved.
#Youre goal is to use this get_data function to retrieve the data
#Then you want to create a plot that shows the average performance of the agent for different credit assignment
#values
#So you should have one plot that has many lines on it. Each line representing the student performance under a different credit assignment value
#You can use the plotting code in run.py for help. But you will have to make minor changes to plot multiple lines.



