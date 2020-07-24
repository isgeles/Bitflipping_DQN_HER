# Bitflipping Experiment with Deep Q-Network and Hindsight Experience Replay

Used Algorithms in Pytorch:
  - Deep Q-Network (DQN)
  - DQN with Hindsight Experience Replay (DQN+HER)


### Environment

Given a sequence of n bits and n target bits, the agent has to flip a bit with every action so that the sequence is equal to the goal sequence.

The episode length is equal to the number of bits, i.e. if all n bits must be flipped for reaching the target, the agent should not make any mistakes to be successful (receive reward at the end of the episode).

A sample unsuccessful episode (for n = 5) could look like this, where with every timestep the agent received a negativ reward of -1:
```
Step  0    Bits: [1 1 0 0 0]   Goal: [0 1 1 0 0]   Success: False
Step  1    Bits: [1 0 0 0 0]   Goal: [0 1 1 0 0]   Success: False
Step  2    Bits: [1 0 0 1 0]   Goal: [0 1 1 0 0]   Success: False
Step  3    Bits: [1 0 1 1 0]   Goal: [0 1 1 0 0]   Success: False
Step  4    Bits: [1 0 0 1 0]   Goal: [0 1 1 0 0]   Success: False
Step  5    Bits: [0 0 0 1 0]   Goal: [0 1 1 0 0]   Success: False
######## DONE in 5 timesteps, success: False ########
```


### Rewards

A reward of -1 is provided with every step the agent takes until all bits are correct. If all bits are correctly flipped, the agent recieves a reward of 0 and the episode ends. 
Thus, the goal of your agent is to flip the wrong bits as quickly as possible. 


### States and Actions

The state space has 2n dimensions (n bits for current and n bits for target sequence). 
The action space is defined by the index of the bit to be flipped, i.e. 0 until n-1 are valid actions to be taken.


### Learning Algorithm

Deep Q-Networks with standard experience replay fail to learn from this environment for n > 13 bits, since the state-space becomes to large to explore.
Therefore I implemented Hindsight Experience Replay (HER) as described in the paper [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) to solve the task up to 40 bits (tested so far).


### Files in this Repository
                    
    .
    ├── trained/                           # stored weights for trained Q-networks 
    ├── Bitflipping_Environment.py         # implementation of environment in openai gym format
    ├── dqn_agent.py                       # agent to interact and learn from environment
    ├── main_ipython.ipynb                 # main code for training and testing the agent in ipython (jupyter) notebook
    ├── main.py                            # main code for training and testing the agent in .py format
    ├── model.py                           # neural network model (in Pytorch)
    ├── replayBuffer.py                    # buffer for experience replay
    └── README.md


### Python Packages
 - numpy
 - random
 - torch
 - collections
 - matplotlib
 - pandas        (for rolling average of success plot)
 - progressbar   (for tracking time during training)
 






