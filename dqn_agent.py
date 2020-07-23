import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import QNetwork
from replayBuffer import ReplayBuffer


class Agent:
    """ Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, batch_size, buffer_size, gamma, lr):
        """ Initialize an Agent.

        @param state_size: (int) dimension of each state (= n)
        @param action_size: (int) dimension of each action (= n), select maximum as action
        @param batch_size: (int) mini-batch size
        @param buffer_size: replay-buffer size
        @param gamma: discount factor
        @param lr: learning rate
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.state_goal_size = 2 * state_size  # state+goal = 2n
        self.action_size = action_size

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.lr = lr

        # Q-Network
        self.qnetwork_local = QNetwork(self.state_goal_size, action_size).to(self.device)
        self.qnetwork_target = QNetwork(self.state_goal_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size)
        
    def store_episode(self, states, actions, rewards, next_states, dones):
        """ Store episode to replay buffer for standard experience replay.

        @param states: (list of dicts) containing 'obs' and 'goal' (is stored as s||g in memory)
        @param actions: list of actions in episode
        @param rewards: list of rewards received in episode
        @param next_states: list of next states (is stored as ns||g in memory)
        @param dones: boolean indicating end of episode
        """
        # normal experience replay, store experiences
        state_goals = [np.concatenate([i['obs'], i['goal']]) for i in states]
        next_state_goals = [np.concatenate([i['obs'], i['goal']]) for i in next_states]
        for (sg, a, r, nsg, d) in zip(state_goals, actions, rewards, next_state_goals, dones):
            self.memory.add(sg, a, r, nsg, d)

    def store_episode_HER(self, states, actions, next_states, replay_strategy='final', k=4):
        """ Store episode with HER samples if replay_strategy is set to 'final', 'future' or 'episode'.

        @param states: (list of dicts) containing 'obs' and 'goal' (is stored as s||g in memory)
        @param actions: list of actions
        @param next_states: list of next states (is stored as ns||g in memory)
        @param replay_strategy: if 'future' HER samples are added to the buffer
        @param k: number of goals in one episode for HER
        """  
        T = len(actions)
        n_bits = len(states[0]['obs'])

        if replay_strategy is 'final':
            # HER 'final' replay strategy ---------------------------------------------------------
            # substitute goal as final state of episode
            goal_her = next_states[-1]['obs']

            for t in range(T):
                state_goal = np.concatenate((states[t]['obs'], goal_her))
                next_state_goal = np.concatenate((next_states[t]['obs'], goal_her))
                # recompute reward and done
                done = np.sum(np.array(next_states[t]['obs']) == np.array(goal_her)) == n_bits
                reward = 0 if done else -1
                self.memory.add(state_goal, actions[t], reward, next_state_goal, done)

        if replay_strategy is 'future':
            # HER 'future' replay strategy ---------------------------------------------------------
            for t in range(T):
                for k in range(k):
                    future_idx = np.random.randint(t, T)  # select random index from future experience in episode
                    # set goal as next_state from future index
                    goal_her = next_states[future_idx]['obs']
                    state_goal = np.concatenate([states[t]['obs'], goal_her])
                    next_state_goal = np.concatenate([next_states[t]['obs'], goal_her])
                    # recompute reward and done
                    done = np.sum(np.array(next_states[t]['obs']) == np.array(goal_her)) == n_bits
                    reward = 0 if done else -1
                    self.memory.add(state_goal, actions[t], reward, next_state_goal, done)

        if replay_strategy is 'episode':
            # HER 'episode' replay strategy ---------------------------------------------------------
            for t in range(T):
                for k in range(k):
                    episode_idx = np.random.randint(0, T)       # select random index from current episode
                    # set goal as random (next) state in episode
                    goal_her = next_states[episode_idx]['obs']
                    state_goal = np.concatenate([states[t]['obs'], goal_her])
                    next_state_goal = np.concatenate([next_states[t]['obs'], goal_her])
                    # recompute reward and done
                    done = np.sum(np.array(next_states[t]['obs']) == np.array(goal_her)) == n_bits
                    reward = 0 if done else -1
                    self.memory.add(state_goal, actions[t], reward, next_state_goal, done)

    def act(self, state_goal, eps=0.):
        """ Returns actions for given state as per current policy

        @param state_goal: (array_like) current state
        @param eps: (float) epsilon, for epsilon-greedy action selection
        @return: (int) action is the index of the bit to flip, value in [0, n-1]
        """
        state_goal = torch.from_numpy(state_goal).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_goal)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        """ Update value parameters using given batch of experience tuples."""
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()

            # compute and minimize the loss

            state_goals, actions, rewards, next_state_goals, dones = experiences

            # update rule
            Q_targets = rewards + \
                        self.gamma * self.qnetwork_target(next_state_goals).max(1)[0].unsqueeze(1) * (1 - dones)

            Q_expected = self.qnetwork_local(state_goals).gather(1, actions)

            # MSE loss
            loss = F.mse_loss(Q_expected, Q_targets)

            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def soft_update(self, local_model, target_model, tau):
        """ Soft update model parameters:
        θ_target = τ*θ_local + (1 - τ)*θ_target

        @param local_model: local pytorch model
        @param target_model: target pytorch model
        @param tau: soft update of target network, 1-tau = polyak coefficient
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)



