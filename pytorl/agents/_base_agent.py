import torch
import torch.nn as nn
from pytorl.networks import io
from pytorl.utils import Setting


"""
this base agent contains:
    1) save/load functionality
    2) tensorboard utilities
"""


class Agent:
    def __init__(self):
        self._optimize_counter = 0
        self._optimize_timer = 0
        self._tensorboard = None
        
    def save_pth(self, obj, path, filename=None, obj_name=None):
        io.save_pth(obj, path, filename=filename, obj_name=obj_name)
        
    def load_pth(self, path, filename=None, obj_name=None):
        io.load_pth(path, filename=filename, obj_name=obj_name)
        
    def optimize_counter(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._optimize_counter += num
        elif pattern == 'set':
            self._optimize_counter = num           
        return self._optimize_counter 

    def optimize_timer(self, pattern=None, num=1):
        assert type(num) == int and num >= 0
        if pattern == 'add':
            self._optimize_timer += num
        elif pattern == 'set':
            self._optimize_timer = num           
        return self._optimize_timer 
        
    @Setting.only_once
    def set_tensorboard(self, obj=None):
        """
        [!]WARNING: should check the legitimacy of num by yourself
        """
        if obj is not None:
            self._tensorboard = obj
        return self._tensorboard