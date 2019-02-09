import torch
import torch.nn as nn
import rl.utils as util


"""
this base agent contains:
    1) save/load functionality
    2) @TODO: tensorboard utilities
"""


class Base_Agent:
    def __init__(self):
        raise NotImplementedError('cannot initialize base agent')
    
    def save_pth(self, obj, path, filename=None, obj_name=None):
        util.save_pth(obj, path, filename=filename, obj_name=obj_name)
        
    def load_pth(self, path, filename=None, obj_name=None):
        util.load_pth(path, filename=filename, obj_name=obj_name)
        
    def add_scalar(self, tensorboard):
        raise NotImplementedError('tensorboard currently not implemented')