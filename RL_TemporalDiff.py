#!/usr/bin/env python
# coding: utf-8

# # Temporal Difference
# Implement the basic TD learning algorithm (i.e. tabular TD(0)) to find a good path (given by a policy) for an agent exploring the environment from a starting point to a reward location in a simple environment.
# 
# Where **S** is the starting location and **E** the end location.

# ## Environment

# In[33]:


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

# In[34]:


def reward(state):
    return 1.0 if WORLD[state.row, state.col] == 'E' else 0.0

reward(initial_state)


# ## Value Function

# In[35]:


def zero_values():
    return np.zeros((WORLD.shape))

def value(state):
    return V[state.row, state.col]


def show_values(decimal_numbers=5):
    expression = "{" + ":.{}".format(decimal_numbers) + "f} " 
    for row in range(V.shape[0]):
        val = "" 
        for col in range(V.shape[1]):
          val += expression.format(V[row, col])
        print(val)
        
V = zero_values()
# print(value(initial_state))
show_values()


# ## Policy
# - Deterministic: (e.g. always take the highest value) 
# - Stochastic: (e.g. random exploration or epsilon-greedy).

# In[36]:


from enum import Enum

class Action(Enum):
    UP = 1
    DOWN = 2
    RIGHT = 3
    LEFT = 4


# In[37]:


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

possible_actions(initial_state)


# In[38]:


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

def next_action(state):
    actions = random.choice(possible_actions(state)) 
    action_to_values = {}
    for action in possible_actions(state):
        action_to_values[action] = value(take_action(state, action))
    max_value = max(action_to_values.values())
    max_actions = [a for a, v in action_to_values.items() if v == max_value]
    return random.choice(max_actions)

# print(take_action(State(5, 4), Action.UP))
# print(next_action(initial_state))
print(next_action(State(5, 3)))


# In[39]:


def end_state(state):
    return WORLD[state.row, state.col] == 'E'


# ## Temporal Difference Learning

# In[40]:


V = zero_values()

alpha = 0.5
gamma = 0.75
episodes = 50
for episode in range(episodes):
    state = initial_state
    exploring = True
    while(exploring):
        action = next_action(state)
        next_state = take_action(state, action)
        learning = alpha * (reward(state) + gamma * value(next_state) - value(state))
        V[state.row, state.col] += learning
        if end_state(state): exploring = False
        state = next_state
show_values(decimal_numbers=10)


# ## Strategy learned

# In[41]:


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
            if is_wall(State(row, col)):
                s_row.append(" ")
            else:
                action = next_action(State(row, col))
                s_row.append(action_encoding[action])
        strategy.append(s_row)
    return strategy

learned_strategy()


# In[42]:


# count how many steps you require to get home


# In[43]:


# Implement Epsylon - Greedy
# Plot how V changes over learning
# Plot how Policy changes


# In[ ]:





# In[ ]:




