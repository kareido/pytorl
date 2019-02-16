import torch
import torch.nn as nn
import pytorl.utils as utils


"""
this base agent contains:
    1) save/load functionality
    2) @TODO: tensorboard utilities
"""


class Agent:
    def __init__(self):
        self._global_timesteps = 0
        self._optimize_timer = 0
        self._tensorboard = None
    
    def save_pth(self, obj, path, filename=None, obj_name=None):
        utils.save_pth(obj, path, filename=filename, obj_name=obj_name)
        
    def load_pth(self, path, filename=None, obj_name=None):
        utils.load_pth(path, filename=filename, obj_name=obj_name)
        
        
    def global_timesteps(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._global_timesteps += num
        elif pattern == 'set':
            self._global_timesteps = num           
        return self._global_timesteps 

    def optimize_timer(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._optimize_timer += num
        elif pattern == 'set':
            self._optimize_timer = num           
        return self._optimize_timer 
        
        
    def set_tensorboard(self, obj=None):
        """
        [!]WARNING: should check the legitimacy of num by yourself
        """
        if obj is not None:
            self._tensorboard = obj
        return self._tensorboard