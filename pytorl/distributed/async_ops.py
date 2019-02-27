import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


"""
[!]WARNING: WORK IN PROGRESS, DO NOT USE
"""


def send_list(tensor_list, dst, group=None, tag=0):
    if group is None: 
        if dist.get_world_size() == 1: return
        _send = lambda tensor: dist.send(tensor, dst, tag=tag)
    else:
        if dist.get_world_size(group) == 1: return
        _send = lambda tensor: dist.send(tensor, dst, group, tag)
    tensor = parameters_to_vector(tensor_list)
    if dist.get_backend() == 'gloo': tensor = tensor.cpu()
    _send(tensor)
    
    
def recv_list(tensor_list, src=None, group=None, tag=0):
    if group is None: 
        if dist.get_world_size() == 1: return
        _recv = lambda tensor: dist.recv(tensor, src, tag=tag)
    else:
        if dist.get_world_size(group) == 1: return
        _recv = lambda tensor: dist.recv(tensor, src, group, tag)
    
    tensor = parameters_to_vector(tensor_list)
    device = 'cuda' if tensor.is_cuda else 'cpu'
    if dist.get_backend() == 'gloo': tensor = tensor.cpu()
    _recv(tensor)
    tensor = tensor.to(device)
    vector_to_parameters(tensor, tensor_list)


def send_model_params(model, dst, rank=-1, group=None, tag=0):
    """
    can use rank to add an optional id info to the footer
    """
    if group is None: 
        if dist.get_world_size() == 1: return
        _send = lambda tensor: dist.send(tensor, dst, tag=tag)
    else:
        if dist.get_world_size(group) == 1: return
        _send = lambda tensor: dist.send(tensor, dst, group, tag)
        
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.cpu()
    tensor_list = tuple(model.state_dict().values())
    for tensor in tensor_list:
        _send(tensor)
    _send(torch.tensor(rank, dtype=torch.float32))
    model.to(device)
        
        
def recv_model_params(model, rank=None, group=None, tag=0):
    """
    can receive sender's rank
    """
    msg = torch.zeros(1)
    if group is None: 
        if dist.get_world_size() == 1: return
        _recv = lambda tensor: dist.recv(tensor, rank, tag=tag)
    else:
        if dist.get_world_size(group) == 1: return
        _recv = lambda tensor: dist.recv(tensor, rank, group, tag)
        
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.cpu()
    tensor_list = tuple(model.state_dict().values())
    for tensor in tensor_list:
        _recv(tensor)
    _recv(msg)
    model.to(device)
    
    return int(msg.item())

    
    