#!/usr/bin/env python
# coding: utf-8


# DQN without target network or experience replay, using constant epsilon value during training.


import tensorflow as tf
import gym
import numpy as np 
import random


# In[ ]:


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


# In[ ]:


class Agent():
    
    def __init__(self):
        self.qModel = QNetwork()
        self.gamma = 0.99
        self.eps = 1
        self.qopt = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    def choose_action(self, state):
        
        cond = random.random() # generate random float between 0 and 1
        
        if cond < self.eps:
            # Choose random action
            action = random.randint(0,1)
            
        else: # Choose greedy action
            Q_s = self.qModel(np.array([state]))
            action = tf.math.argmax(Q_s, axis=1)
            action = action.numpy()[0]
            
        return action
    
    def train(self, state, action, reward, next_state): # train from an episode of experience
        
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
        
        return error


# In[ ]:


agent = Agent()
env = gym.make("CartPole-v1")
cum_rew_train = []
cum_rew_test = []


# In[ ]:


"""
 ---------------
# TRAINING CELL #
 --------------- 
"""

def main():
    
    for e in range(300):
        
        obs, info = env.reset()
        done = False
        score = 0
        
        while not done:
            
            action = agent.choose_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            score += reward
            done = terminated or truncated
            
            error = agent.train(obs, action, reward, next_obs)
            
            obs = next_obs
            
        cum_rew_train.append(score)
        print(f"Episode {e} score is {score}")
        
main()


# In[ ]:


# Change agent epsilon
agent.eps = 0.01


# In[ ]:


"""
 --------------
# TESTING CELL #
 --------------
"""

def main():
    
    for e in range(300):
        
        obs, info = env.reset()
        done = False
        score = 0
        
        while not done:
            
            action = agent.choose_action(obs)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            score += reward
            done = terminated or truncated
    
        # No weight updates in testing mode    
        
            obs = next_obs
            
        cum_rew_test.append(score)
        print(f"Episode {e} score is {score}")
        
main()


# In[ ]:


import matplotlib.pyplot as plt
# plot reward graphs

fig = plt.gcf()
plt.title("DQN Cumulative Reward for 'CartPole-v1' with lr=1e-3")
plt.plot(cum_rew_train, label='training')
plt.plot(cum_rew_test, label='testing')
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Reward")
plt.legend()


# In[ ]:




