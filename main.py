import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import progressbar as pb           # tracking time while training

from Bitflipping_Environment import BitFlippingEnv
from dqn_agent import Agent

DEFAULT_PARAMS = {
    'n_bits': 35,                 # n bits to flip in environment (n corresponding target bits)
    'seed': 0,                    # random seed for environment, torch, numpy, random packages

    'eps': 0.2,                   # probability of random action, 'epsilon-greedy' policy
    'buffer_size': int(1e6),      # replay-buffer size
    'batch_size': 64,             # mini-batch size
    'gamma': 0.98,                # discount factor
    'tau': 0.05,                  # soft update of target network, 1-tau = polyak coefficient
    'lr': 0.001,                  # learning rate

    # training setup
    # HINT: for higher number of bits, strategy 'final' works better
    'replay_strategy': 'final',   # 'none' (ignore HER), 'final','future','episode' for HER
    'n_epochs': 200,              # number of epochs, HER paper: 200 epochs (i.e. maximum of 8e6 timesteps)
    'n_cycles': 50,               # number of cycles per epoch, HER paper: 50 cycles
    'n_episodes': 16,             # number of episodes per cycle, HER paper: 16 episodes
    'n_optim': 40,                # number of optimization steps every cycle, HER paper: 40 steps
}


def set_seeds(seed: int = 0):
    """ Set random seed for all used packages (numpy, torch, random).

    @param seed: random seed to be set, default is 0.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pass


def test_bitflipping(n_bits, n_episodes=5, policy=None, render=False):
    """ Test bitflipping environment, optionally with agent as policy and rendering (printing).

    @param n_bits: number of bits
    @param n_episodes: number of episodes to test
    @param policy: agent to act according to current policy
    @param render: print states of environment
    @return: average success-rate of number of episodes
    """
    env = BitFlippingEnv(n_bits)
    success = []

    for e in range(n_episodes):
        state, _, _, _ = env.reset()
        if render:
            env.render()
        for t in range(1000):

            if policy is None:
                action = np.random.randint(0, n_bits)
            else:
                state_goal = np.concatenate([state['obs'], state['goal']])
                action = policy.act(state_goal, eps=0)

            state, reward, done, info = env.step(action)
            if render:
                env.render()
            if done:
                break
        success.append(int(info))
    return np.mean(success)


def train(n_bits, agent):
    """ Train DQN agent, with option for HER.

    @param n_bits: number of bits in bitflip environment
    @param agent: DQN agent to learn
    @return: list of success from every episode
    """
    print("Training DQN on Bitflipping with", n_bits, "bits for", DEFAULT_PARAMS['n_epochs'], "epochs...")

    # widget bar to display progress during training
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=DEFAULT_PARAMS['n_epochs']).start()

    env = BitFlippingEnv(n_bits)
    success = []
    eps = DEFAULT_PARAMS['eps']
    for i_epoch in range(1, DEFAULT_PARAMS['n_epochs'] + 1):
        for i_cycle in range(DEFAULT_PARAMS['n_cycles']):
            for i_episode in range(DEFAULT_PARAMS['n_episodes']):
                state, _, _, _ = env.reset()
                state_ep, act_ep, reward_ep, next_state_ep, done_ep = [], [], [], [], []
                for t in range(1000):
                    state_goal = np.concatenate([state['obs'], state['goal']])
                    action = agent.act(state_goal, eps)
                    next_state, reward, done, info = env.step(action)

                    # save current transition of episode
                    state_ep.append(state.copy())
                    act_ep.append(action)
                    reward_ep.append(reward)
                    next_state_ep.append(next_state.copy())
                    done_ep.append(done)

                    state = next_state
                    if done:
                        break
                success.append(int(info))
                # for standard experience replay
                agent.store_episode(state_ep, act_ep, reward_ep, next_state_ep, done_ep)
                # HER: save additional goals
                # if not info:  # use HER only if unsuccessful episode
                agent.store_episode_HER(state_ep, act_ep, next_state_ep,
                                        replay_strategy=DEFAULT_PARAMS['replay_strategy'])

            # optimize and soft update of networks
            for _ in range(DEFAULT_PARAMS['n_optim']):
                agent.learn()
            agent.soft_update(agent.qnetwork_local, agent.qnetwork_target, DEFAULT_PARAMS['tau'])

        # stop training earlier
        if np.mean(success[-50:]) > 0.98:
            print("\n learning done")
            break

        if i_epoch % (DEFAULT_PARAMS['n_cycles'] / 10) == 0:
            print('\rEpoch {} \t Success: {:.4f}'.format(i_epoch, np.mean(success[-50:])))
        timer.update(i_epoch)
    timer.finish()
    return success


def main():
    """ Main: sets random seed and trains the agent as defined in DEFAULT_PARAMS and saves trained networks.
    """
    set_seeds(DEFAULT_PARAMS['seed'])
    # test_bitflipping(DEFAULT_PARAMS['n_bits'], n_episodes=3, render=True)  # testing random actions in env

    agent = Agent(DEFAULT_PARAMS['n_bits'], DEFAULT_PARAMS['n_bits'], DEFAULT_PARAMS['batch_size'],
                  DEFAULT_PARAMS['buffer_size'], DEFAULT_PARAMS['gamma'], DEFAULT_PARAMS['lr'])

    success = train(DEFAULT_PARAMS['n_bits'], agent)

    # saving trained network and results
    torch.save(agent.qnetwork_local.state_dict(), './trained/checkpoint_' + str(DEFAULT_PARAMS['n_bits']) + 'bits.pth')
    # np.savetxt('./trained/success_'+str(DEFAULT_PARAMS['n_bits'])+'.csv', success, delimiter=',')

    # Plot rolling average of success
    N = 300
    rolling_avg = pd.Series(success).rolling(window=N).mean().iloc[N - 1:].values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rolling_avg)), rolling_avg)
    plt.ylabel('Success-rate')
    plt.xlabel('Episode #')
    plt.show()

    # load the weights from file
    #agent.qnetwork_local.load_state_dict(torch.load('./trained/checkpoint_30bits.pth'))

    print("Testing agent for 100 episodes, success-rate: ",
          test_bitflipping(DEFAULT_PARAMS['n_bits'], n_episodes=100, policy=agent)*100, "%")


if __name__ == '__main__':
    main()


