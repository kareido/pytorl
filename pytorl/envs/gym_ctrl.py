import gym
import numpy as np
import torch
from collections import deque
from ._base_env import MetaEnv


class _Gym1DWrapper(gym.Wrapper, metaclass=MetaEnv):
    """
    this class is a wrapper of classic control problems in gym environment that helps:
        1) preprocessing the observation:
               i. stack multiple observations for neural net input.
        2) directly sample an action
        3) initialize episode (and frames stack) with random actions
        4) misc methods and other methods required to be implemented by metaclass MetaEnv

    example:
        env = make_gym('CartPole-v1', render=True)

    another important criterion is that only convert an obj to tensor or make it downsampled iff.
    imminent necessay, otherwise, try not to make conversion which will confuse you latter

    """
    def __init__(self, env, render):
        super(_Gym1DWrapper, self).__init__(env)
        self.render_flag = render
        # frame initialization
        self.prev_observ = None
        self.curr_observ = None
        self.curr_state = None
        # frame stack buffer
        self.buffer = deque([], maxlen=1)


    def set_frames_action(self, num=1):
        self.frames_action('set', num)


    def set_frames_stack(self, num=1):
        self.frames_stack('set', num)
        self.buffer = deque([], maxlen=num)


    def num_actions(self):
        return self.action_space.n


    def observ_shape(self):
        return self.observation_space.shape[0]


    def _feed_buffer(self):
        # let buffer save transformed 1-D frame
        self.buffer.append(self.curr_observ)
        self.prev_observ = self.curr_observ


    def _preprocessing(self):
        assert len(self.buffer) == self.frames_stack() == self.buffer.maxlen
        observs_tensor = torch.tensor(self.buffer, dtype=torch.float32)
        # .unsqueeze(0) here to make it ready for input
        return observs_tensor.unsqueeze(0)


    def sample(self):
        return self.action_space.sample()


    def state(self):
        self.curr_state = self._preprocessing()
        return self.curr_state


    def reset(self):
        # reset episodic attributions
        self.reset_statistics('episodic')
        self.buffer.clear()
        self.prev_observ = self.env.reset()
        init_frames = self.buffer.maxlen
        assert init_frames > 0, 'minimum buffer length should be 1'
        get_action = self.sample
        for _ in range(init_frames):
            self.curr_observ, reward, done, _ = self.env.step(get_action())
            if done: print('EXCEPTION: done received during reset()', flush=True)
            self._feed_buffer()
        # statistics
        self.global_frames('add', init_frames + 1)
        self.global_episodes('add')
        self.episodic_frames('add', init_frames + 1)


    def refresh(self):
        # state and status
        self.prev_observ = None
        self.curr_state = None
        # frame stack buffer
        self.buffer = deque([], maxlen=self.frames_stack())
        self.reset_statistics('all')


    def step(self, action):
        if isinstance(action, torch.Tensor): action = action.item()
        _action_reward = 0
        for _ in range(self.frames_action()):
#             self.prev_observ = self.curr_observ
            self.curr_observ, reward, done, info = self.env.step(action)
            _action_reward += reward
            self._feed_buffer()
            if self.render_flag: self.render()
            self.global_frames('add')
            self.episodic_frames('add')
            if done: break
#         self._feed_buffer()
        _action_reward = np.sign(_action_reward)

        self.episodic_reward('add', _action_reward)
        self.action_reward('set', _action_reward)
        return self.curr_observ, self.action_reward(), done, info



def make_ctrl_env(env_name, render=False):
    orig_env = gym.make(env_name)
    wrapped_env = _Gym1DWrapper(orig_env, render)
    return wrapped_env

