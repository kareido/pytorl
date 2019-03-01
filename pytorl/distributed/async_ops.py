import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def isend_list(tensor_list, dst, group=None, tag=0):
    if group is None: 
        if dist.get_world_size() == 1: return
        _isend = lambda tensor: dist.send(tensor, dst, tag=tag)
    else:
        if dist.get_world_size(group) == 1: return
        _isend = lambda tensor: dist.isend(tensor, dst, group, tag)
    tensor = parameters_to_vector(tensor_list)
    if dist.get_backend() == 'gloo': tensor = tensor.cpu()
    return _isend(tensor)
    
    
    