import torch
import torch.nn as nn


"""
this base agent contains:
    1) save/load functionality
    2) @TODO: tensorboard utilities
"""


class Agent:
    def __init__(self):
        self._optimize_counter = 0
        self._optimize_timer = 0
        self._tensorboard = None
        
        
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
        
        
    def set_tensorboard(self, obj=None):
        """
        [!]WARNING: should check the legitimacy of num by yourself
        """
        if obj is not None:
            self._tensorboard = obj
        return self._tensorboard