import torch
import torch.distributed as dist
from .setup import *


def all_reduce_mean(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)
        tensor.div_(get_world_size())


def all_reduce_sum(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.SUM)


def all_reduce_max(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=reduce_op.MAX)
        

def all_reduce_min(tensor_list):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=reduce_op.MAX)
        tensor.neg_()


def broadcast(tensor_list, src):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    if get_world_size() == 1:
        return
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


def all_gather_cat(tensor_list, cat_dim=0):
    if isinstance(tensor_list, torch.Tensor):
        raise ValueError('tensor_list should be list of tensors')
    world_size = get_world_size()
    if world_size == 1:
        return tensor_list
    result_list = []
    for tensor in tensor_list:
        gather_list = [tensor.new(tensor.size()) for _ in range(world_size)]
        dist.all_gather(gather_list, tensor)
        result_list.append(torch.cat(gather_list, cat_dim))
    return result_list
