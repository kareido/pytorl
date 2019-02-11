import gym
import numpy as np
import torch
from collections import deque


"""
this class is a wrapper of original gym environment that helps:
    1) make the observation in the shape of C * H * W and converts 
       it along with the reward to torch tensors.
    2) count steps when step() is called.
    3) directly sample an action
    4) multiple frames stack for neural net input.
    5) initialize episode (and frames stack) with noop, random, fire ops
    6) one-life per episode mode

example:
    resize = T.Compose([T.ToPILImage(),
                    T.Grayscale(1), 
                    T.Resize((84, 84)), 
                    T.ToTensor()])
    env = get_env('Breakout-v0', resize, render=True)
    agent.set_frame_stack(num_frames=4, stack_init_mode='fire')
    agent.set_single_life_mode(False)    
    
"""
class _CommonWrapper(gym.Wrapper):
    def __init__(self, env, tsfm, render):
        super(_CommonWrapper, self).__init__(env)
        self.tsfm = tsfm
        self.render = render

        self.curr_step_count = 0
        self.total_step_count = 0
        # frame stack
        self.num_frames = None
        self.stack_init_mode = None
        self.frame_stack = None
        # single life mode
        self.only_single_life = False
        self.lives = 0
    
    
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.tolist()
        if self.frame_stack is not None:
            # check if frame stack is cold
            if len(self.frame_stack) < self.num_frames:
                self._init_episode(mode = self.stack_init_mode, 
                                   steps = self.num_frames - 1)
            else:
                self.frame_stack.popleft()
                
        ob, reward, done, info = self.env.step(action)
        ob = self.transform_ob(ob)
        
        self.curr_step_count += 1
        self.total_step_count += 1
        
        if self.frame_stack is not None:
            self.frame_stack.append(ob)
            assert len(self.frame_stack) == self.num_frames
            ret_ob = self.transform_ob(self.frame_stack)
            
        if self.only_single_life:
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
        
        if self.render:
            self.env.render()
            
        return ret_ob.unsqueeze(0), self.reward(reward), done, info

    
    def reset(self, init_mode=None, init_steps=1):
        self.curr_step_count = 0
        self.env.reset()
        if self.frame_stack is not None:
            self.frame_stack.clear()
        if init_mode:
            _init_episode(mode = init_mode, 
                          steps = init_steps)
        return self.transform_ob()

    
    def transform_ob(self, ob=None):
        if ob is None:
            ob = self.env.render(mode='rgb_array')
        if isinstance(ob, deque):
            ob_tensor = torch.cat(tuple(ob))
            return ob_tensor
        
        elif isinstance(ob, (list, tuple)):
            ob_tensor = torch.cat(ob)
            return ob_tensor
        
        elif isinstance(ob, (np.ndarray,)):
            ob_tensor = torch.tensor(ob)
        elif isinstance(ob, (torch.Tensor,)):
            ob_tensor = ob.to('cpu')
        else:
            raise ValueError('unsupported transform_ob type')
        return self.tsfm(ob_tensor)
    
    
    # using clipped reward
    def reward(self, reward):
#         print(np.sign(reward), flush=True)
        return torch.tensor(np.sign(reward))

    
    def sample(self):
        return torch.tensor(self.action_space.sample())
    
    
    def set_frame_stack(self, num_frames=4, stack_init_mode='random'):
        assert type(num_frames) == int
        if num_frames <= 1:
            raise ValueError('invalid num_frames specified')
        if num_frames == 1:
            print('warning: agent only has current '
                  'transform_ob (num_frames = 1)', flush=True)
            return
        self.num_frames = num_frames
        self.stack_init_mode = stack_init_mode
        self.frame_stack = deque([])
        
        
    def _init_episode(self, mode, steps):
        assert mode in {'random', 'noop', 'fire'}
        assert steps >= 1, 'illegal steps'
        if mode == 'random':
            action = self.sample()
        elif mode == 'noop':
            assert self.unwrapped.get_action_meanings()[0] == 'NOOP'
            action = 0
        elif mode == 'fire':
            assert self.unwrapped.get_action_meanings()[1] == 'FIRE'
            action = 1
        if self.frame_stack is not None:
            assert not self.frame_stack, 'stack not empty'
        for _ in range(steps):
            ob, _, done, _ = self.env.step(action)
            ob = self.transform_ob(ob)
            if self.frame_stack is not None:
                self.frame_stack.append(ob)
                if done:
                    while len(self.frame_stack) < steps:
                        self.frame_stack.append(ob)
                    break
            if done:
                break
        if done:
            print('warning: already done '
                  'during initializing episode', flush=True)
        return done
                
        
    def set_single_life_mode(self, status=True):
        self.only_single_life = status
        if status:
            print('SINGLE LIFE MODE: ON', flush=True)
        else:
            print('SINGLE LIFE MODE: OFF', flush=True)
    

    
def get_env(env_name, tsfm, render=False):
    orig_env = gym.make(env_name)
    wrapped_env = _CommonWrapper(orig_env, tsfm, render)
    return wrapped_env

