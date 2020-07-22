import numpy as np

class BitFlippingEnv():
    def __init__(self, n: int, seed: int = None):
        """Bitflipping environment in the format of openai gym environments with reset(), step(), render().
        The environment starts with a random state and goal of 'n' each.
        Within the episode of 'n' timesteps, the episode is successful if all 'n' bits are correct flipped.

        @param n: (int) number of bits
        @param seed: random seed for numpy
        """
        if n <= 0:
            raise ValueError("n must be positive integer")
        # number of bits
        self.n = n

        # action is the index of the bit to flip
        self.action_space = np.arange(n)

        self.timestep = 0
        if seed is not None:
            self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset environment."""
        self.obs = np.random.randint(2, size=self.n)    # random initial obs
        self.goal = np.random.randint(2, size=self.n)   # random goal
        self.reward = 0.                                # reward, only 0 if success
        self.done = False                               # episode finished (i.e. 'n' timesteps passed or success)
        self.info = False                               # if successful or not (boolean)
        self.timestep = 0
        # goal should be different to initial obs
        while np.sum(self.obs == self.goal) == self.n:
            self.goal = np.random.randint(2, size=self.n)

        self.state = {'obs': self.obs.copy(), 'goal': self.goal}
        return self.state, self.reward, self.done, self.info

    def step(self, action):
        """Perform action and update state."""
        # flipping the bit at position 'action'
        self.obs[action] = 1 - self.obs[action]
        self.timestep += 1
        self.reward = -1.

        # success if n bits are correct
        if np.sum(self.obs == self.goal) == self.n:
            self.reward = 0.
            self.info = True
            self.done = True
        else:
            self.reward = -1.
            self.info = False
            self.done = False

        if self.timestep == self.n:
            self.done = True

        self.nextstate = {'obs': self.obs.copy(), 'goal': self.goal}
        return self.nextstate, self.reward, self.done, self.info

    def render(self, obs=None):
        """ Render state (obs and goal)."""
        if obs is None:
            obs = self.obs
        print("Step {:2}    Bits: {}   Goal: {}   Success: {}".format(self.timestep, obs, self.goal, self.info))
        if self.done:
            print("DONE in", self.timestep, "timesteps, success:", self.info, "\n")



