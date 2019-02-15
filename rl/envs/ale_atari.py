import gym
import numpy as np
import torch
from collections import deque
from ._base_env import MetaEnv


class _AtariWrapper(gym.Wrapper, metaclass=MetaEnv):
    """
    this class is a wrapper of original ale atari in gym environment that helps:
        1) preprocessing the observation:
             i. find max value of each corresponding pixel between 2 consecutive frame
            ii. make observation state in the shape of C * H * W and converts it to torch tensors.
           iii. stack multiple frames for neural net input.
        2) directly sample an action
        3) initialize episode (and frames stack) with noop, random, fire ops
        4) single-life per episode mode
        5) misc methods and other methods required to be implemented by metaclass MetaEnv

    example:
        self.resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(1), 
                        T.Resize((84, 84)), 
                        T.ToTensor()])
        env = make_atari('Breakout-v0', resize, render=True)
        agent.frame_stack('set', 4)
        agent.single_life('set', False)

    another important criterion is that only convert an obj to tensor or make it downsampled iff. 
    imminent necessay, otherwise, try not to make conversion which will confuse you latter
    
    """
    def __init__(self, env, tsfm, render):
        super(_AtariWrapper, self).__init__(env)
        self.tsfm = tsfm
        self.render = render
        # frame initialization
        self.episodic_init_action = None
        self.episodic_init_steps = 0
        self.buffer_init_action = None
        # frame stack buffer
        self.buffer = None
        # single life mode
        self.only_single_life = False
        self.lives = 0
    
    
    def set_frames_action(self, num=1):
        self.frames_action('set', num)
        
    def set_frames_stack(self, num=1, op='NOOP'):
        self.frames_stack('set', num)
        self.buffer = deque([], maxlen=num)
        self.buffer_init_action = op
        
    def set_episodic_init(self, op='FIRE', steps=1):
        self.episodic_init_action = op
        self.episodic_init_steps = steps
    
    
    
    
    

def make_atari_env(env_name, tsfm, render=False):
    orig_env = gym.make(env_name)
    wrapped_env = _AtariWrapper(orig_env, tsfm, render)
    return wrapped_env

