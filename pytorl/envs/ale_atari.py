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
        resize = T.Compose([T.ToPILImage(),
                        T.Grayscale(1), 
                        T.Resize((84, 84)), 
                        T.ToTensor()])
        env = make_atari('Breakout-v0', resize, render=True)
        env.set_episodic_init('FIRE')
        env.set_frames_stack(frame_stack)
        env.set_single_life(True)
        env.set_frames_action(4)

    another important criterion is that only convert an obj to tensor or make it downsampled iff. 
    imminent necessay, otherwise, try not to make conversion which will confuse you latter
    
    """
    def __init__(self, env, tsfm, render):
        super(_AtariWrapper, self).__init__(env)
        self.tsfm = tsfm
        self.render_flag = render
        # frame initialization
        self.episodic_init_action = 'RANDOM'
        self.episodic_init_frames = 0
        # state and status
        self.prev_observ = None
        self.curr_observ = None
        self._new_observ_buffer = deque([], maxlen=1)
        self.curr_state = None
        # frame stack buffer
        self.buffer = deque([], maxlen=1)
        # single life mode
        self.single_life = False
        self.lives = 0
    
    
    def set_frames_action(self, num=1):
        self.frames_action('set', num)
        self._new_observ_buffer = deque([], maxlen=num)
        
        
    def set_frames_stack(self, num=1):
        self.frames_stack('set', num)
        self.buffer = deque([], maxlen=num)
        
        
    def set_episodic_init(self, op='RANDOM', frames=1):
        assert type(frames) == int and frames >= 1
        if op is None: op = 'RANDOM'
        self.episodic_init_action = op
        self.episodic_init_frames = frames
    

    def set_single_life(self, flag=True):
        self.single_life = flag
        
    
    def num_actions(self):
        return self.action_space.n
    
    
    def _feed_buffer(self):
        # deflickering previous and current observation
        if len(self._new_observ_buffer) == 0:
            max_pooled_observ = self.curr_observ
        else:
            max_pooled_observ = np.max(np.stack(self._new_observ_buffer), axis=0)
        deflickered_observ = np.maximum(self.prev_observ, max_pooled_observ)
        # let buffer save transformed 2-D frame
        encoded_frame = self.tsfm(deflickered_observ)
        self.buffer.append(encoded_frame)
        self.prev_observ = self.curr_observ
    
    
    def _preprocessing(self):
        assert len(self.buffer) == self.frames_stack() == self.buffer.maxlen
        observs_tensor = torch.cat(tuple(self.buffer))
        return observs_tensor.unsqueeze(0)
        
        
    def sample(self):
        return self.action_space.sample()
    
    def state(self):
        self.curr_state = self._preprocessing()
        return self.curr_state
    
    def _get_init_action(self, op):
        assert op in {'RANDOM', 'NOOP', 'FIRE'}
        if op == 'RANDOM':
            wrapper = self.sample
        elif op == 'NOOP':
            assert self.unwrapped.get_action_meanings()[0] == 'NOOP'
            wrapper = lambda: 0
        elif op == 'FIRE':
            assert self.unwrapped.get_action_meanings()[1] == 'FIRE'
            wrapper = lambda: 1
        return wrapper
    
    
    def reset(self):
        # reset episodic attributions
        self.reset_statistics('episodic')
        self.buffer.clear()
        self._new_observ_buffer.clear()
        self.prev_observ = self.env.reset()
        if self.render_flag: self.render()
        init_frames = max(self.episodic_init_frames, self.buffer.maxlen)
        assert init_frames > 0, 'minimum buffer length should be 1'
        get_action = self._get_init_action(self.episodic_init_action)
        for _ in range(init_frames):
            self.curr_observ, reward, done, _ = self.env.step(get_action())
            self._new_observ_buffer.append(self.curr_observ)
            if done: print('EXCEPTION: done received during reset()', flush=True)
            self._feed_buffer()
        # statistics
        self.global_frames('add', init_frames + 1)
        self.global_episodes('add')
        self.episodic_frames('add', init_frames + 1)
        
        
    def refresh(self):
        # state and status
#         self.prev_observ = None
        self._new_observ_buffer = deque([], maxlen=self.frames_action())
        self.curr_state = None
        # frame stack buffer
        self.buffer = deque([], maxlen=self.frames_stack())
        self.reset_statistics('all')
        
    def step(self, action):
        if isinstance(action, torch.Tensor): action = action.item()
        _action_reward = 0
        for _ in range(self.frames_action()):
            self.curr_observ, reward, done, info = self.env.step(action)
            self._new_observ_buffer.append(self.curr_observ)
#             self._feed_buffer()
            _action_reward += reward
            if self.render_flag: self.render()
            self.global_frames('add')
            self.episodic_frames('add')
            if done: break
        self._feed_buffer()
        _action_reward = np.sign(_action_reward)
        if self.single_life:
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True
            self.lives = lives
        
        self.episodic_reward('add', _action_reward)
        self.action_reward('set', _action_reward)
        return self.curr_observ, self.action_reward(), done, info
        
    

def make_atari_env(env_name, tsfm, render=False):
    orig_env = gym.make(env_name)
    wrapped_env = _AtariWrapper(orig_env, tsfm, render)
    return wrapped_env

