import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import pandas as pd
import progressbar as pb           # tracking time while training

from Bitflipping_Environment import BitFlippingEnv
from dqn_agent import Agent

DEFAULT_PARAMS = {
    'n_bits': 50,                 # n bits to flip in environment (n corresponding target bits)
    'seed': 0,                    # random seed for environment, torch, numpy, random packages

    'eps': 0.2,                   # probability of random action, 'epsilon-greedy' policy
    'buffer_size': int(1e6),      # replay-buffer size
    'batch_size': 64,             # mini-batch size
    'gamma': 0.95,                # discount factor
    'tau': 0.05,                  # soft update of target network, 1-tau = polyak coefficient
    'lr': 0.001,                  # learning rate

    # training setup
    'replay_strategy': 'future',  # 'none' for vanilla ddpg, 'future' for HER
    'n_epochs': 200,              # number of epochs, HER paper: 200 epochs (i.e. maximum of 8e6 timesteps)
    'n_cycles': 50,               # number of cycles per epoch, HER paper: 50 cycles
    'n_episodes': 16,             # number of episodes per cycle, HER paper: 16 episodes
    'n_optim': 40,                # number of optimization steps every cycle, HER paper: 40 steps
}


def set_seeds(seed: int = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    pass


def test_bitflipping(n, n_episodes=5, policy=None, render=False):
    env = BitFlippingEnv(n)
    success = []

    for e in range(n_episodes):
        state, _, _, _ = env.reset()
        if render:
            env.render()
        for t in range(n):

            if policy is None:
                action = np.random.randint(0, n)
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


def train(n, agent):
    print("Training DQN on Bitflipping with", n, "bits for", DEFAULT_PARAMS['n_epochs'], "epochs...")

    # widget bar to display progress during training
    widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
    timer = pb.ProgressBar(widgets=widget, maxval=DEFAULT_PARAMS['n_epochs']).start()

    env = BitFlippingEnv(n)
    success = []
    eps = DEFAULT_PARAMS['eps']
    for i_epoch in range(1, DEFAULT_PARAMS['n_epochs'] + 1):
        for i_cycle in range(DEFAULT_PARAMS['n_cycles']):
            for i_episode in range(DEFAULT_PARAMS['n_episodes']):
                state, _, _, _ = env.reset()
                state_ep, act_ep, reward_ep, next_state_ep, done_ep = [], [], [], [], []
                for t in range(DEFAULT_PARAMS['n_bits']):
                    state_goal = np.concatenate([state['obs'], state['goal']])
                    action = agent.act(state_goal, eps)
                    next_state, reward, done, info = env.step(action)

                    state_ep.append(state.copy())
                    act_ep.append(action)
                    reward_ep.append(reward)
                    next_state_ep.append(next_state.copy())
                    done_ep.append(done)

                    success.append(int(info))
                    if done:
                        break
                    state = next_state
                agent.store_episode(state_ep, act_ep, reward_ep, next_state_ep, done_ep)

                # HER additional goals
                agent.store_episode_HER(state_ep, act_ep, reward_ep, next_state_ep, done_ep,
                                        replay_strategy=DEFAULT_PARAMS['replay_strategy'])

            for _ in range(DEFAULT_PARAMS['n_optim']):
                agent.learn()
        agent.soft_update(agent.qnetwork_local, agent.qnetwork_target, agent.tau)

        # stop training
        if np.mean(success[-10:]) > 0.999:
            print("\n learning done")
            break

        if i_epoch % (DEFAULT_PARAMS['n_cycles'] / 10) == 0:
            print('\rEpoch {} \t Success: {:.4f}'.format(i_epoch, np.mean(success[-10:])))

        timer.update(i_epoch)
    timer.finish()
    return success




def main():

    set_seeds(DEFAULT_PARAMS['seed'])
    # test_bitflipping(DEFAULT_PARAMS['n_bits'], n_episodes=3, render=True)  # testing random actions in env

    agent = Agent(DEFAULT_PARAMS['n_bits'], DEFAULT_PARAMS['n_bits'],
                  DEFAULT_PARAMS['batch_size'], DEFAULT_PARAMS['buffer_size'], DEFAULT_PARAMS['gamma'],
                  DEFAULT_PARAMS['tau'], DEFAULT_PARAMS['lr'])

    success = train(DEFAULT_PARAMS['n_bits'], agent)

    # saving trained network and results
    torch.save(agent.qnetwork_local.state_dict(), './trained/checkpoint_' + str(DEFAULT_PARAMS['n_bits']) + 'bits.pth')
    np.savetxt('./trained/success_'+str(DEFAULT_PARAMS['n_bits'])+'.csv', success, delimiter=',')

    # Plot rolling average of success
    N = 300
    rolling_avg = pd.Series(success).rolling(window=N).mean().iloc[N - 1:].values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(rolling_avg)), rolling_avg)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


    # load the weights from file
    #agent.qnetwork_local.load_state_dict(torch.load('./trained/checkpoint_30bits.pth'))

    print("Testing agent for 100 episodes, success-rate: ", test_bitflipping(DEFAULT_PARAMS['n_bits'], n_episodes=100,
                                                                        policy=agent)*100, "%")



if __name__ == '__main__':
    main()


