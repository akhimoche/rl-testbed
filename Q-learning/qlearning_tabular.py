#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym 
import numpy as np
np.set_printoptions(suppress=True)

env = gym.make("FrozenLake-v1", is_slippery=True)
action_space = env.action_space.n
state_space = env.observation_space.n

class Agent():
    
    def __init__(self):
        self.qtable = np.zeros((state_space, action_space), dtype=float)
        self.gamma = 0.95
        self.eps_max = 1
        self.eps_min = 0.001
        self.decay_rate = 0.001
        self.lr = 0.8
        
    def choose_action(self, state, ep_num):
        
        epsilon = self.eps_min + (self.eps_max-self.eps_min)*np.exp(-self.decay_rate*ep_num)
        cond = np.random.random()
        
        if cond < epsilon:
            
            action = env.action_space.sample()
            
        else:
        
            action = np.argmax(self.qtable[state, :])
            
        return action
    
    def update(self, state, action, reward, next_state):
        
        prediction = self.qtable[state, action]
        target = reward + self.gamma * np.max(self.qtable[next_state, :])*(1-done) # The (1-done) condition makes learning much less stable
        update = target - prediction
        self.qtable[state, action] = self.qtable[state, action] + self.lr * update
        


# In[2]:


"""
-----------------------
~    TRAINING CELL    ~
-----------------------

"""

agent = Agent()
cum_rew = []

for e in range(25000):
    #print(e, end="\r")
    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        
        action = agent.choose_action(obs, e)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        done = terminated or truncated
        
        agent.update(obs, action, reward, next_obs)
        
        obs = next_obs
    
    cum_rew.append(total_reward)

print(agent.qtable)


# In[3]:


"""
-------------------
~    TEST CELL    ~
-------------------

"""


test_rew = []
agent.eps_max =0.001
for e in range(10000):
    #print(e, end="\r")
    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        
        action = agent.choose_action(obs, e)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        done = terminated or truncated
        
        obs = next_obs
    
    test_rew.append(total_reward)

print(sum(test_rew)/len(test_rew))


# In[ ]:




