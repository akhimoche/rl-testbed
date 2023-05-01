#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np 
import random
from collections import deque


# In[2]:


num_of_episodes = 1000


# In[3]:


class QNetwork(tf.keras.Model):
    
    def __init__(self):
        
        super().__init__()
        # Shared layers for policy and value function networks
        self.layer1 = tf.keras.layers.Dense(1024, activation = 'relu')
        self.layer2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.out = tf.keras.layers.Dense(2, activation = None) # ... from state to action probabilities for pi
        
    def call(self, state):
        
        x = tf.convert_to_tensor(state)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.out(x)
        
        return out


# In[4]:


class Agent():
    
    def __init__(self):
        self.gamma = 0.99
        self.eps_max = 1
        self.eps_min = 0.01
        self.decay_rate = 0.01 # number of episodes over which epsilon is decayed
        
        self.qModel = QNetwork()
        self.qopt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
    def choose_action(self, state, e):
        
        cond = random.random() # generate random float between 0 and 1
        
        eps = self.eps_min + (self.eps_max-self.eps_min)*np.exp(-self.decay_rate*e)
        
        if cond < eps:
            # Choose random action
            action = random.randint(0,1)
            
        else: # Choose greedy action
            Q_s = self.qModel(np.array([state]))
            action = tf.math.argmax(Q_s, axis=1)
            action = action.numpy()[0]
            
        return action
    
    def train(self, buffer): # train from buffer
        
        random.shuffle(buffer) # shuffle the buffer
        
        for state, action, reward, next_state, done in buffer:

            state = np.array([state])
            next_state = np.array([next_state])

            with tf.GradientTape() as tape:

                # Get prediction
                a =  action # action chosen to calculate Q(s,a)
                Q_s = self.qModel(state, training=True) # select the value of the Q value associated with pair s,a
                Q_s_a = Q_s[0][a]

                # Get target
                Q_sp = self.qModel(next_state, training=True)
                max_Q = tf.reduce_max(Q_sp, axis=1)
                target = reward + self.gamma*max_Q

                # Compute error
                error = (target - Q_s_a)**2

            grads = tape.gradient(error, self.qModel.trainable_variables)
            self.qopt.apply_gradients(zip(grads, self.qModel.trainable_variables))
            


# In[5]:


agent = Agent()
env = gym.make("CartPole-v1")
cum_rew_train = []
cum_rew_test = []


# In[6]:


"""
 ---------------
# TRAINING CELL #
 --------------- 
"""

def main():
    
    # Create a summary writer for TensorBoard
    writer = tf.summary.create_file_writer("logs/")
    
    for e in range(num_of_episodes):
        
        obs, info = env.reset()
        done = False
        score = 0
        buffer = []
        
        while not done:
            
            action = agent.choose_action(obs, e)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            score += reward
            done = terminated or truncated
            
            buffer.append((obs, action, reward, next_obs, done))
            
            if done:
                agent.train(buffer)
            
            obs = next_obs
            
        cum_rew_train.append(score)
        print(f"Episode {e} score is {score}")
        
        # Tensorboard log stuff
#         with writer.as_default():
#             tf.summary.scalar(name="cumulative_reward", data=score, step=e)
#             dqn_variables = agent.qModel.trainable_variables
#             for i, var in enumerate(dqn_variables):
#                 tf.summary.histogram(name=f"dqn_variable_{i}", data=var, step=e)
#                 if len(var.shape) == 2:
#                     # Log the weights of each layer
#                     kernel = var[:, :]
#                     tf.summary.histogram(name=f"layer_{i}_weights", data=kernel, step=e)
#             writer.flush()
        
main()


# In[7]:


# Enter below command to anaconda terminal to visualise results of training:
# tensorboard --logdir=.
# (You need to navigate to log file first)


# In[8]:


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


# In[9]:


""" Statistics """

mean_reward = np.mean(cum_rew_train)
deviation = np.std(cum_rew_train)

print(f"Cumulative reward mean is {mean_reward} +/- {deviation}")


# In[10]:


array = cum_rew_train
aver = moving_average(array, 50)

import matplotlib.pyplot as plt
fig = plt.gcf()
plt.title("DQN w/ ER Cumulative Reward graph against moving average")
plt.plot(array, label='training')
plt.plot(aver, label='moving average')
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Reward")
plt.legend()


# In[ ]:




