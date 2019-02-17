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


def init_network(network, pretrained_path=None, filename=None, obj_name=None):
    """
    load a pretrained model

    Args:
        network: the network which is going to be fed with the pretrained .pth
        pretrained_path: supposed to be a string containing the path and filename if specifying 
            just the path without filename, it is also ok so long as the filename is correctly set.
            if pretrained_path is invalid(e.g. file not found or not a string), this function will 
            return immediately without load
        filename: supposed to be set if path is literally just the path without the file name.
        obj_name: the string that will be the name of the network obj, if not set, the loading 
            process will be silent.
    
    """    
    if type(pretrained_path) != str: return
    elif filename is not None: pretrained_path = os.path.join(pretrained_path, filename)
    if not os.path.isfile(pretrained_path): return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        device = 'cuda:%s' % torch.cuda.current_decice()
        print('mapping cuda to [%s]' % device, flush=True)
    checkpoint = torch.load(
        pretrained_path,
        map_location = device
    )
    network.load_state_dict(checkpoint)
    if obj_name is not None:
        print('[%s] successfully loaded from [%s]' % (
                obj_name, os.path.abs,_path(pretrained_path)), flush=True)