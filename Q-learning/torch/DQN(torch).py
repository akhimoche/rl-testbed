#!/usr/bin/env python
# coding: utf-8

# In[9]:


import gym
import numpy as np
import random
from collections import deque

import torch
from torch import nn
from torch.optim import Adam


# In[10]:


env = gym.make('CartPole-v1')

action_size = env.action_space.n
observation_size = env.observation_space.shape[0]


# In[11]:


class QNetwork(nn.Module):
    
    def __init__(self):
        
        super(QNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.pipe = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size), # Get Q values for each action in a state
        )
        
    def forward(self, state):
        
        logits = self.pipe(state)
        
        return logits

    


# In[12]:


""" QNetwork Unit Test """

# Should return a 2x1 tensor

#X = torch.rand(4)

# model = QNetwork()
# logits = model(X)
# print(logits)

# print(X)
# print(torch.max(X,dim=0)[1])
# print(torch.max(X,dim=0)[1].item())


# In[13]:


class Agent():
    
    def __init__(self):
        self.qModel = QNetwork()
        self.gamma = 0.99
        self.eps_max = 1
        self.eps_min = 0.01
        self.decay_rate = 0.01 # number of episodes over which epsilon is decayed
        self.qopt = Adam(self.qModel.parameters(), lr=1e-4)
        
    def choose_action(self, state, e):
        
        cond = random.random() # generate random float between 0 and 1
        
        eps = self.eps_min + (self.eps_max-self.eps_min)*np.exp(-self.decay_rate*e)
        
        if cond < eps:
            # Choose random action
            action = random.randint(0,1)
            
        else: # Choose greedy action
            state = torch.from_numpy(state).float()
            Q_s = self.qModel(state)
            action = torch.max(Q_s,dim=0)[1]
            action = action.item()
            
        return action
    
    def train(self, state, action, reward, next_state, done): # train from an episode of experience
        
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float()
            
        # Get prediction
        a =  action # action chosen to calculate Q(s,a)
        Q_s = self.qModel(state) # select the value of the Q value associated with pair s,a
        Q_s_a = Q_s[a]
            
        # Get target
        Q_sp = self.qModel(next_state)
        max_Q = torch.max(Q_sp)
        target = reward + self.gamma*max_Q*done
            
        # Compute error
        error = (target - Q_s_a)**2
            
        self.qopt.zero_grad()
        error.backward()
        self.qopt.step()
        
        return error


# In[14]:


agent = Agent()
env = gym.make("CartPole-v1")
cum_rew = []
num_of_episodes = 300


# In[15]:


"""
 ---------------
# TRAINING CELL #
 --------------- 
"""

def main():
    
    for e in range(num_of_episodes):
        
        obs, info = env.reset()
        done = False
        score = 0
        
        while not done:
            
            action = agent.choose_action(obs, e)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            score += reward
            done = terminated or truncated
            
            agent.train(obs, action, reward, next_obs, 1-terminated)
            
            obs = next_obs
            
        cum_rew.append(score)
        print(f"Episode {e} score is {score}")
        
main()


# In[16]:


import matplotlib.pyplot as plt

def moving_average(arr, window_size):
    """
    Calculate moving average of a NumPy array.

    Parameters:
        arr (numpy.ndarray): Input array.
        window_size (int): Size of moving window.

    Returns:
        numpy.ndarray: Array of moving averages.

    """
    # Create a list to hold the moving averages
    moving_averages = []

    # Iterate through the array
    for i in range(len(arr)):
        # Calculate the start and end indices of the moving window
        start_index = max(0, i - window_size + 1)
        end_index = i + 1

        # Get the subset of the array within the moving window
        subset = arr[start_index:end_index]

        # Calculate the average of the subset and append it to the list of moving averages
        moving_averages.append(np.mean(subset))

    # Convert the list of moving averages to a NumPy array and return it
    return np.array(moving_averages)


# In[ ]:


array = cum_rew
aver = moving_average(array, 50)

fig = plt.gcf()
plt.title("DQN Cumulative Reward graph against moving average")
plt.plot(array, label='training')
plt.plot(aver, label='moving average')
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Reward")
plt.legend()


# In[ ]:




