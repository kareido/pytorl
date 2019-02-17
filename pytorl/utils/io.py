import os
import torch
import torch.nn as nn


def save_pth(obj, path, filename=None, obj_name=None):
    """
    save an obj on the disk as .pth

    Args:
        obj: a serializable object, if specified as an nn obj (e.g. a neural net), the state_dict will 
            be fetched from it and saved eventually.
        path: supposed to be a string containing the path and filename if specifying just the path 
            without filename, it is also ok so long as the filename is correctly set.
        filename: supposed to be set if path is literally just the path without the file name.
        obj_name: the string that will be the name of the saved obj, if not set, the saving process 
            will be silent.
    """
    if isinstance(obj, nn.Module):
        obj = obj.state_dict()
    if filename:
        path = os.path.join(path, filename)
    filedir = os.path.dirname(path)
    # check path existence
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    torch.save(obj, path)
    if obj_name:
        print('[%s] successfully saved at [%s]' % (
                obj_name, os.path.abspath(path)), flush=True)
        

def load_pth(path, filename=None, obj_name=None):
    """
    save an obj on the disk as .pth

    Args:
        path: supposed to be a string containing the path and filename if specifying just the 
            path without filename, it is also ok so long as the filename is correctly set.
        filename: supposed to be set if path is literally just the path without the file name.
        obj_name: the string that will be the name of the loaded obj, if not set, the loading 
            process will be silent.

    Return:
        loaded: the loaded object
    """
    if filename:
        path = os.path.join(path, filename)
    loaded = torch.load(path)
    if obj_name:
        print('[%s] successfully loaded from [%s]' % (
                obj_name, os.path.abs,_path(path)), flush=True)
    return loaded


def init_network(conf):
    network = getattr(models, conf.arch)()
    network.change_output_classes(conf.num_classes)
    rank, world_size = alt_dist.get_rank(), alt_dist.get_world_size()
    if rank == 0:
        print('\narchitecture: [%s]' % conf.arch, flush=True)
    
    checkpoint = torch.load(
        conf.pretrain_path,
        map_location = 'cuda:%d' % torch.cuda.current_device()
    )
    try:
        network.load_state_dict(checkpoint['network'])
#         alt_dist.barrier()
        print('rank [%s/%s] resumed from best ckpt: [%s]' % (rank, world_size, conf.pretrain_path), flush=True)
    except KeyError:
        if rank == 0:
            print("KEY ERROR FOUND AS THERE IS NO KEY NAMED [network] IN [checkpoint]", flush=True)
            print('trying to execute network.load_state_dict(checkpoint, strict=True)...', flush=True)
        try:
            network.load_state_dict(checkpoint, strict=True)
#             alt_dist.barrier()
            print('rank [%s/%s] resumed from best ckpt: [%s]' % (rank, world_size, conf.pretrain_path), flush=True)            
        except:
            if rank == 0:
                print('network.load_state_dict(checkpoint, strict=True) FAILED AS KEYS MISMATCH', flush=True)
                print('trying to execute network.load_state_dict(checkpoint, strict=False)...', flush=True)
            try:
                network.load_state_dict(checkpoint, strict=False)
#                 alt_dist.barrier()
                print('rank [%s/%s] resumed from best ckpt: [%s]' % (rank, world_size, conf.pretrain_path), flush=True)
            except:
                raise ValueError('unhandled checkpoint file structure')
    
    alt_dist.barrier()
    if rank == 0:
        print('network on all ranks (total num: [%s]) initialization completed\n' % world_size, flush=True)
    return network