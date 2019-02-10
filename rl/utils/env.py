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
        self.curr_step_count = 0
        self.total_step_count = 0
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.curr_step_count += 1
        self.total_step_count += 1
        return self.observation(observation), self.reward(reward), done, info

    def reset(self, **kwargs):
        self.curr_step_count = 0
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def observation(self, observation=None):
        if observation is None:
            observation = self.env.render(mode='rgb_array')
        ob_tensor = torch.tensor(observation, device=self.device)
        return self.tsfm(ob_tensor)
    
    def reward(self, reward):
        return torch.tensor(reward).to(self.device)

    def sample(self):
        return torch.tensor(self.action_space.sample()).to(self.device)
    
    
def get_env(env_name, tsfm, device):
    orig_env = gym.make(env_name)
    wrapped_env = _TensorWrapper(orig_env, tsfm, device)
    return wrapped_env


def get_stacked_ob_func(env, stack_num):
    def stacked_ob_func(action):
        ret = env.observation()
        for _ in range(stack_num):
            ob, _, done, _ = env.step(action)
                ret = torch.cat((ret, ob))
            if done:
                while ret.shape[0] < stack_num:
                    ret = torch.cat((ret, ob))
                break
        # reform ret as (N, C, H, W)
        return ret.unsqueeze(0)   
    
    return stacked_ob_func