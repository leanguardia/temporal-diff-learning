#!/usr/bin/env python
# coding: utf-8

# # Temporal Difference
# Implement the basic TD learning algorithm (i.e. tabular TD(0)) to find a good path (given by a policy) for an agent exploring the environment from a starting point to a reward location in a simple environment.
# 
# Where **S** is the starting location and **E** the end location.

# ## Environment

# In[1]:


import numpy as np
from collections import namedtuple

WORLD = np.array([
  [' ', ' ', ' ', ' ', ' ', ' ', ' '],
  [' ', 'W', 'W', 'W', 'W', ' ', ' '], 
  [' ', 'W', 'E', ' ', 'W', ' ', ' '],
  [' ', ' ', 'W', ' ', 'W', 'W', ' '],
  [' ', ' ', 'W', ' ', ' ', ' ', ' '],
  ['S', 'W', ' ', ' ', ' ', ' ', ' '],
 ])
world_height = WORLD.shape[0]
world_width = WORLD.shape[1]

State = namedtuple('State', 'row col')
initial_state = State(5, 0)


# ## Reward Signal

# In[2]:


def reward(state):
    return 1.0 if WORLD[state.row, state.col] == 'E' else 0.0

reward(initial_state)


# ## Value Function

# In[3]:


def zero_values():
    return np.zeros((WORLD.shape))

def value(state):
    return V[state.row, state.col]


def show_values(decimal_numbers=5):
    cell_value = "{" + ":.{}".format(decimal_numbers) + "f} " 
    for row in range(V.shape[0]):
        val = "" 
        for col in range(V.shape[1]):
          val += cell_value.format(V[row, col])
        print(val)
        
V = zero_values()
# print(value(initial_state))
show_values()


# ## Policy
# - Deterministic: (e.g. always take the highest value) 
# - Stochastic: (e.g. random exploration or epsilon-greedy).

# In[4]:


from enum import Enum

class Action(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


# In[5]:


def is_wall(position):
    return WORLD[position.row, position.col] == 'W'
def can_go_left(state):
    return state.col > 0 and not is_wall(State(state.row, state.col-1))
def can_go_right(state):
    return state.col < world_width-1 and not is_wall(State(state.row, state.col+1))
def can_go_up(state):
    return state.row > 0 and not is_wall(State(state.row-1, state.col))
def can_go_down(state):
    return state.row < world_height-1 and not is_wall(State(state.row+1, state.col))
def possible_actions(state):
    actions = []
    if can_go_left(state):  actions.append(Action.LEFT)
    if can_go_right(state): actions.append(Action.RIGHT)
    if can_go_up(state):    actions.append(Action.UP)
    if can_go_down(state):  actions.append(Action.DOWN)
    return actions
def end_state(state):
    return WORLD[state.row, state.col] == 'E'

possible_actions(initial_state)


# In[6]:


import random

def go_up(state):
    return State(state.row-1, state.col)
def go_down(state):
    return State(state.row+1, state.col)
def go_left(state):
    return State(state.row, state.col-1)
def go_right(state):
    return State(state.row, state.col+1)

def take_action(prev_state, action):
    if action is Action.UP: return go_up(prev_state)
    if action is Action.DOWN: return go_down(prev_state)
    if action is Action.LEFT: return go_left(prev_state)
    if action is Action.RIGHT: return go_right(prev_state)
    
def converged(steps_per_episode, num_of_evaluations=10):
    if (len(steps_per_episode) <= num_of_evaluations):
        return False
    else:
        tail = steps_per_episode[-num_of_evaluations:]
        last = steps_per_episode[-1]
        return all(last == steps for steps in tail)
    
def random_exploration(state):
    return random.choice(possible_actions(state)) 

def maximum_value(state):
    actions = random.choice(possible_actions(state)) 
    action_to_values = {}
    for action in possible_actions(state):
        action_to_values[action] = value(take_action(state, action))
    max_value = max(action_to_values.values())
    max_actions = [a for a, v in action_to_values.items() if v == max_value]
    return random.choice(max_actions)

def epsylon_greedy(state, current_episode, episodes_to_exploitation):
    epsylon = 1 - (current_episode / episodes_to_exploitation)
    epsylon = max(0, epsylon)
    rand_number = random.random()
    if (rand_number < epsylon):
        action = random_exploration(state)
    else:
        action = maximum_value(state)
    return action

def next_action(state, policy, episode):
    if policy == "EPSY":
        action = epsylon_greedy(state, episode, episodes_to_exploitation=50)
    elif policy == "MAX":
        action = maximum_value(state)
    elif policy == "RAND":
        action = random_exploration(state)
    else: print("PI: Not implemented")
    return action

# print(maximum_value(State(5, 3)))
print(epsylon_greedy(State(4, 0), 0, 1))


# ## Temporal Difference Learning
# 
# **Parameters:**
# 
# Alpha - Learning Rate [0, 1] - How much we update our value estimate at each sample
# 
# Gamma - Discount Rate [0, 1]

# In[7]:


V = zero_values()

policy = "EPSY"
# policy = "MAX"
# policy = "RAND"

alpha = 0.75
gamma = 0.9
max_episodes = 500

steps_per_episode = []
episode = 0
while(not converged(steps_per_episode, num_of_evaluations=15) and episode < max_episodes):
    state = initial_state
    interacting = True
    step = 0
    while(interacting):
        action = next_action(state, policy, episode)
        next_state = take_action(state, action)
        prediction_error = alpha * (reward(state) + gamma * value(next_state) - value(state))
        V[state.row, state.col] += prediction_error
        if end_state(state):
            interacting = False
            episode += 1
        else:
            state = next_state
            step +=1
    steps_per_episode.append(step)
show_values(decimal_numbers=5)


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

number_of_episodes = len(steps_per_episode)
print("Episodes required:", number_of_episodes)
max_steps = np.max(steps_per_episode)
print("Maximum steps:", max_steps)
min_steps = np.min(steps_per_episode)
print("Minimum steps:", min_steps)
fig = plt.figure(figsize=(8, 5));
plt.plot(range(number_of_episodes), steps_per_episode)
print(steps_per_episode)


# In[9]:


fig, axs = plt.subplots(figsize=(7, 6))
im = axs.imshow(V, cmap="GnBu")
fig.colorbar(im, fraction=0.04, pad=0.03)
plt.grid()
plt.show()


# ## Strategy learned

# In[13]:


action_encoding = {
    Action.UP: 'U',
    Action.DOWN: 'D',
    Action.LEFT: 'L',
    Action.RIGHT: 'R',
}

def learned_strategy():
    strategy = []
    for row in range(world_height):
        s_row = []
        for col in range(world_width):
            if is_wall(State(row, col)): s_row.append(" ")
            else:
                action = maximum_value(State(row, col))
                s_row.append(action_encoding[action])
        strategy.append(s_row)
    return strategy

learned_strategy()


# In[ ]:





# In[12]:


index = np.array(range(75))
epsilon = np.maximum(0, (-0.02 * index) + 1)
fig, ax = plt.subplots(figsize=(8, 5))
ax.text(7, 0.35, "Explore", fontsize=14)
ax.text(30, 0.70, "Exploit", fontsize=14)
ax.plot(index, epsilon, color="k")
ax.fill_between(index, epsilon, color="green", alpha="0.3")
ax.fill_between(index, epsilon, np.max(epsilon), alpha="0.3")
plt.xlabel("Episodes")
plt.ylabel("Epsilon")
plt.show()


# In[ ]:




