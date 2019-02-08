import gym
import numpy as np
import torch


"""
this class is a wrapper of original gym environment that helps the observation
be in the shape of C * H * W and converts observation along with the reward to
torch tensors

example:
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1), 
                    T.Resize((84, 84)), 
                    T.ToTensor()])
    env = get_env('Breakout-v0', resize, device)
"""
class _TensorWrapper(gym.Wrapper):
    def __init__(self, env, tsfm, device):
        super(_TensorWrapper, self).__init__(env)
        self.device = device
        self.tsfm = tsfm
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), self.reward(reward), done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation):
        return self.tsfm(observation).to(self.device)
    
    def reward(self, reward):
        return torch.tensor(reward, dtype=torch.float32).to(self.device)

    
def get_env(env_name, tsfm, device):
    orig_env = gym.make(env_name)
    wrapped_env = _TensorWrapper(orig_env, tsfm, device)
    return wrapped_env
