#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import gym
import numpy as np 
import random
from collections import deque
import cProfile
import matplotlib.pyplot as plt


# In[ ]:



num_of_episodes = 500

env = gym.make('CartPole-v1')

action_size = env.action_space.n


# In[ ]:


class QNetwork(tf.keras.Model):
    
    def __init__(self):
        
        
        super().__init__()
        # Shared layers for policy and value function networks
        self.layer1 = tf.keras.layers.Dense(512, activation = 'relu')
        self.layer2 = tf.keras.layers.Dense(512, activation = 'relu')
        self.out = tf.keras.layers.Dense(action_size, activation = None) # ... 2 outputs to move left or right
        
    def call(self, state):
        
        x = tf.convert_to_tensor(state)
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.out(x)
        
        return out


# In[ ]:


class Agent():
    
    def __init__(self):
        self.gamma = 0.99
        self.eps_max = 1
        self.eps_min = 0.01
        self.decay_rate = 0.01 # number of episodes over which epsilon is decayed
        self.buffer_size = 10000
        self.batch_size = 32
        self.update_steps = 100
        self.timestep = 0
        
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
            action = random.randint(0,action_size-1)
            
        else: # Choose greedy action
            Q_s = self.qModel(np.array([state]))
            action = tf.math.argmax(Q_s, axis=1)
            action = action.numpy()[0]
            
        return action
    
    def update_target_network(self):
        q_weights = self.qModel.get_weights()
        self.targetModel.set_weights(q_weights)
    
    def train(self): # train from buffer
        
        if len(self.buffer)<self.batch_size:
            return
        
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        sample = random.sample(self.buffer, k=self.batch_size)
        
        for experience in sample:
            
            state, action, reward, next_state, done = experience
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
            
        with tf.GradientTape() as tape:
            
            Q_s = self.qModel(states,training=True)
            Q_s_a = tf.gather(Q_s, actions, batch_dims=1)
            
            Q_sp = self.targetModel(next_states, training=True)
            max_Q = tf.reduce_max(Q_sp, axis=1)
            target = rewards + self.gamma * max_Q * (1 - dones_tensor)
            loss = tf.math.reduce_mean((target - Q_s_a) ** 2)
        
        self.qopt.minimize(loss, self.qModel.trainable_variables, tape=tape)
        
        if self.timestep % self.update_steps == 0:
            self.update_target_network()
        
        return loss


# In[ ]:


agent = Agent()
cum_rew_train = []


# In[ ]:


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
            
            obs = np.atleast_1d(obs)
            
            action = agent.choose_action(obs, e)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_obs = np.atleast_1d(next_obs)
            
            score += reward
            done = terminated or truncated
            
            agent.buffer.append((obs, action, reward, next_obs, terminated))
            
            agent.train()
            
            agent.timestep += 1
            obs = next_obs
            
        cum_rew_train.append(score)
        print(f"Episode {e} score is {score}")
        

        
if __name__ == "__main__":
    cProfile.run('main()')


# In[ ]:


def moving_average(arr, window_size):
    
    moving_averages = []

    for i in range(len(arr)):
        
        start_index = max(0, i - window_size + 1)
        end_index = i + 1
        subset = arr[start_index:end_index]
        moving_averages.append(np.mean(subset))

    return np.array(moving_averages)



""" Statistics """

mean_reward = np.mean(cum_rew_train)
deviation = np.std(cum_rew_train)

print(f"Cumulative reward mean is: {mean_reward} +/- {deviation}")


array = cum_rew_train
aver = moving_average(array, 50)

fig = plt.gcf()
plt.title(f'DQN: α={agent.lr}, γ={agent.gamma}, ε_dec={agent.decay_rate}, batch={agent.batch_size}, C={agent.update_steps}')
fig.set_size_inches(10.5, 10.5)
plt.plot(array, label='training')
plt.plot(aver, label='moving average')
plt.xlabel("Episode Number")
plt.ylabel("Cumulative Reward")
plt.legend()


# In[ ]:


env.close()


# In[ ]:




