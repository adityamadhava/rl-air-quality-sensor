import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os


class SensorTransmissionEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.lam = 0.1
        self.B = 10
        self.eta = 2
        self.Delta = 3

        base = os.path.dirname(os.path.abspath(__file__))
        self.P_air = np.load(os.path.join(base, 'air.npy'))
        self.P_solar = np.load(os.path.join(base, 'solar.npy'))

        self.n_theta = 51
        self.observation_space = spaces.MultiDiscrete([51, self.B + 1, 51, 51])
        self.action_space = spaces.Discrete(3)

        self.max_steps = 288
        self.state = None
        self._t = 0

    def valid_actions(self, b):
        return [0, 1, 2] if b >= self.eta else [0]

    def step(self, action):
        theta, b, theta_hat, max_val = self.state

        delta = np.random.choice(self.Delta + 1, p=self.P_solar)

        ack = False
        if action in (1, 2) and b >= self.eta:
            tx_val = theta if action == 1 else max_val
            ack = np.random.rand() < self.lam
            new_theta_hat = tx_val if ack else theta_hat
            new_b = min(self.B, b - self.eta + delta)
        else:
            new_theta_hat = theta_hat
            new_b = min(self.B, b + delta)

        # loss uses new_theta_hat (decided after transmission)
        t_r = theta * 0.02
        th_r = new_theta_hat * 0.02
        diff = abs(t_r - th_r)
        loss = diff if t_r <= th_r else 1.5 * diff

        new_theta = np.random.choice(self.n_theta, p=self.P_air[theta])
        new_max_val = new_theta if ack else max(max_val, new_theta)

        self._t += 1
        self.state = (new_theta, new_b, new_theta_hat, new_max_val)
        return self.state, -loss, False, self._t >= self.max_steps, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        theta = np.random.randint(0, self.n_theta)
        b = np.random.randint(0, self.B + 1)
        theta_hat = np.random.randint(0, self.n_theta)
        # max_val starts at current theta — no transmission history at episode start
        max_val = theta

        self._t = 0
        self.state = (theta, b, theta_hat, max_val)
        return self.state, {}

    def render(self):
        # Used to graphics. NOT NEEDED FOR THIS ASSIGNMENT.
        pass
