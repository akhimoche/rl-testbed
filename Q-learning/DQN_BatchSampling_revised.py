#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np 
import random
from collections import deque
import cProfile


# Sample efficient version of grad accumulation taking advantage of tensorflow parallelisation in the forward pass. 


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# In[3]:


num_of_episodes = 1000


# In[4]:


class QNetwork(tf.keras.Model):
    
    def __init__(self):
        
        super().__init__()
        # Shared layers for policy and value function networks
        self.layer1 = tf.keras.layers.Dense(512, activation = 'relu')
        self.layer2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.out = tf.keras.layers.Dense(2, activation = None) # ... 2 outputs to move left or right
        
    def call(self, state):
        
        x = tf.convert_to_tensor(state)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.out(x)
        
        return out


# In[5]:


class Agent():
    
    def __init__(self):
        self.gamma = 0.99
        self.eps_max = 1
        self.eps_min = 0.01
        self.decay_rate = 0.01 # number of episodes over which epsilon is decayed
        self.buffer_size = 1000
        self.batch_size = 32
        self.update_steps = 100
        
        self.buffer = deque(maxlen=self.buffer_size)
        self.qModel = QNetwork()
        self.targetModel = QNetwork()
        self.targetModel.set_weights(self.qModel.get_weights())
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
    
    def update_target_network(self):
        self.targetModel.set_weights(self.qModel.get_weights())
    
    @tf.function
    def train(self, buffer): # train from buffer
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for experience in buffer:
            
            state, action, reward, next_state, done = experience
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        with tf.GradientTape() as tape:
            
            Q_s = self.qModel(states,training=True)
            Q_s_a = tf.gather(Q_s, actions, batch_dims=1)
            
            Q_sp = self.qModel(next_states, training=True)
            max_Q = tf.reduce_max(Q_sp, axis=1)
            target = rewards + 0.99 * max_Q
            loss = tf.math.reduce_mean((target - Q_s_a) ** 2)
        
        self.qopt.minimize(loss, self.qModel.trainable_variables, tape=tape)
        
        return loss
            


# In[6]:


agent = Agent()
env = gym.make("CartPole-v1")
cum_rew_train = []


# In[7]:


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
        

        
if __name__ == "__main__":
    cProfile.run('main()')


# In[9]:


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


# In[10]:


""" Statistics """

mean_reward = np.mean(cum_rew_train)
deviation = np.std(cum_rew_train)

print(f"Cumulative reward mean is: {mean_reward} +/- {deviation}")


# In[11]:


array = cum_rew_train
aver = moving_average(array, 50)

import matplotlib.pyplot as plt
fig = plt.gcf()
plt.title("DQN New Batch Method Cumulative Reward graph against moving average")
plt.plot(array, label='training')
plt.plot(aver, label='moving average')
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Reward")
plt.legend()


# In[ ]:





# In[ ]:




