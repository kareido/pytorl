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

    def sample(self):
        return torch.tensor(self.action_space.sample()).to(self.device)
    
    
def get_env(env_name, tsfm, device):
    orig_env = gym.make(env_name)
    wrapped_env = _TensorWrapper(orig_env, tsfm, device)
    return wrapped_env


def get_stacked_ob_func(env, stack_num):
    def stacked_ob_func(action):
        ret = None
        for _ in range(stack_num):
            ob, _, done, _ = env.step(action)
            if ret is None:
                ret = ob.clone()
            else:
                ret = torch.cat((ret, ob))
            if done:
                while ret.shape[0] < stack_num:
                    ret = torch.cat((ret, ob))
                break
        # reform ret as (N, C, H, W)
        return ret.unsqueeze(0)   
    
    return stacked_ob_func