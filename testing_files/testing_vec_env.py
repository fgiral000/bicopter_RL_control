import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_util import make_vec_env


class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0, 1, (1,))
        self.action_space = spaces.Discrete(2)
        self.a = 45

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, False, False, {}

    def set_a(self, new_a):
        self.a = new_a


vec_env = make_vec_env(MyEnv)
print("Original value: ", vec_env.get_attr("a"))
vec_env.env_method("set_a", 150)
print("Setted value: ",vec_env.get_attr("a"))