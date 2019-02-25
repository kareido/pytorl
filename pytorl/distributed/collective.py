import torch
import torch.distributed as dist
from torch.nn.utils import parameters_to_vector, vector_to_parameters


"""
[!]WARNING: WIP, DO NOT USE
"""


def all_reduce_mean(tensor_list, group=None, async_op=False):
    handler_list = []
    if group is None: 
        _allreduce = lambda tensor: dist.all_reduce(tensor, reduce_op.SUM, async_op=async_op)
    else:
        _allreduce = lambda tensor: dist.recv(tensor, reduce_op.SUM, group, async_op)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        handler = _allreduce(tensor)
        handler_list.append(handler)
        tensor.div_(get_world_size())
    if async_op: return handler_list


def all_reduce_sum(tensor_list, group=None, async_op=False):
    handler_list = []
    if group is None: 
        _allreduce = lambda tensor: dist.all_reduce(tensor, reduce_op.SUM, async_op=async_op)
    else:
        _allreduce = lambda tensor: dist.recv(tensor, reduce_op.SUM, group, async_op)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        handler = _allreduce(tensor)
        handler_list.append(handler)
    if async_op: return handler_list

    
def all_reduce_max(tensor_list, group=None, async_op=False):
    handler_list = []
    if group is None: 
        _allreduce = lambda tensor: dist.all_reduce(tensor, reduce_op.MAX, async_op=async_op)
    else:
        _allreduce = lambda tensor: dist.recv(tensor, reduce_op.MAX, group, async_op)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        handler = dist._allreduce(tensor)
        handler_list.append(handler)
    if async_op: return handler_list        

    
def all_reduce_min(tensor_list, group=None, async_op=False):
    handler_list = []
    if group is None: 
        _allreduce = lambda tensor: dist.all_reduce(tensor, reduce_op.MAX, async_op=async_op)
    else:
        _allreduce = lambda tensor: dist.recv(tensor, reduce_op.MAX, group, async_op)
    if isinstance(tensor_list, torch.Tensor):
        raise TypeError('tensor_list should be list of tensors')
    if get_world_size() == 1: return
    for tensor in tensor_list:
        tensor.neg_()
        handler = dist._allreduce(tensor)
        handler_list.append(handler)
        tensor.neg_()
    if async_op: return handler_list

    
def broadcast_list(tensor_list, src, group=None, async_op=False):
    if group is None: 
        _broadcast = lambda tensor: dist.broadcast(tensor, src, async_op=async_op)
    else:
        _broadcast = lambda tensor: dist.broadcast(tensor, src, group, async_op)
    if get_world_size() == 1: return
    tensor = parameters_to_vector(tensor_list)
    device = 'cuda' if tensor.is_cuda else 'cpu'
    if get_backend() == 'gloo': tensor = tensor.cpu()
    if get_backend() == 'nccl': src_tensor = src.cuda()
    handler = _broadcast(tensor)
    tensor = tensor.to(device)
    if async_op: return handler
    
    
    