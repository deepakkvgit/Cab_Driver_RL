# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():
    
    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =  [(j,i) for j in range(m) for i in range(m) if i != j]
        self.action_space.append((0,0))
        self.state_space = [[i,j,k] for i in range(m) for j in range(t) for k in range(d)]
        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros(36, dtype = int).tolist()
        state_encod[state[0]] = 1
        state_encod[m + state[1]] = 1
        state_encod[m + t + state[2]] = 1
                
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests == 0:
            return [0], [(0,0)]

        if requests >15:
            requests =15
                
        

        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) # (0,0) is not considered as customer request
        
        actions = [self.action_space[i] for i in possible_actions_index]
      
        #actions.append([0,0])

        return possible_actions_index,actions   
    
    
    def new_time_day(self, hour, day, travel_time):

        if (hour + travel_time) < 24:
            hour = hour + travel_time
        else:
            hour = (hour + travel_time) % 24 
            num_days = (hour + travel_time) // 24
            day = (day + num_days) % 7

        return int(hour), int(day)

    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        
        # Get the next state and the various time durations
        next_state, inactive_time, travel_time_to_start_loc, travel_time = self.next_state_func(
            state, action, Time_matrix)

        # Calculate the reward based on the different time durations
        rewards = self.reward_func(inactive_time, travel_time_to_start_loc, travel_time)
        
        total_time = inactive_time + travel_time_to_start_loc + travel_time
        
        return rewards, next_state, total_time
    

    def reward_func(self, inactive_time, travel_time_to_start_loc, travel_time):

        reward = R * (travel_time) - C * (travel_time + travel_time_to_start_loc + inactive_time)
        
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        start_loc = action[0]
        end_loc = action[1]
        current_loc = state[0]
        hour_of_day = state[1]
        day_of_week = state[2] 
        
        total_time = 0
        inactive_time = 0
        travel_time_to_start_loc = 0
        travel_time = 0
        
        if ((start_loc == 0) and (end_loc == 0)):
            # request declined
            inactive_time = 1
        elif (start_loc == current_loc):
            # get travel time from start loc to end loc
            travel_time = Time_matrix[start_loc, end_loc, hour_of_day, day_of_week]
        else: #(current_loc != start_loc)
            # travel time to the start loc from current loc
            travel_time_to_start_loc = Time_matrix[current_loc, start_loc, hour_of_day, day_of_week]
            # new time and day after driver reaches the start loc
            new_hour, new_day = self.new_time_day(hour_of_day, day_of_week, travel_time_to_start_loc)
            # get travel time from start loc to end loc
            travel_time = Time_matrix[start_loc, end_loc, new_hour, new_day]
            # new time and day after driver reaches the end loc
        

        #calculate total time
        total_time = inactive_time + travel_time + travel_time_to_start_loc
        
        #calcuate new time and day
        new_hour, new_day = self.new_time_day(hour_of_day, day_of_week, total_time)
        #set new state
        state = [end_loc, new_hour, new_day]
        return state, inactive_time, travel_time_to_start_loc, travel_time

    def reset(self):
        state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, state_init
