import argparse
import yaml
import torch.nn.functional as F
import torch.optim as optim


class _DotConfig(dict):
    """
    override __getattr__ method in dict class to make config dict dot-accessible
    
    [!]WARNING: can only be used to get attribution but cannot modefy original config value
    """ 
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            return super().__getattr__(key)
        if isinstance(value, dict):
            return _DotConfig(value)
        return value

    def __setattr__(self, key, value):
        raise UserWarning('cannot override config entry')



""" [!]DEPRECATED
THIS METHOD IS NO LONGER IN USE
"""
def _get_config(filename=None, default='config.yaml'):
    """
    Args:
        filename: is used to parse config filename directly in the program if not specifed, config 
            file will be parsed via argparse
        default: default config parser argument
    """
    if filename is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-cfg', type=str, default=default)
        _cfg_name = parser.parse_args().config
    else:
        _cfg_name = filename
    
    assert type(_cfg_name) == str, 'invalid config filename specified (not a string type)'
    assert _cfg_name.rsplit('.', 1)[-1] == 'yaml', 'unspported config file type yet (not .yaml)'
    
    with open(_cfg_name, 'r') as _cfg_f:
        _raw_cfg = yaml.load(_cfg_f)
    
    return _DotConfig(_raw_cfg)



class ConfigReader:
    """
    public interface for getting config conents
    Args:
        filename: is used to parse config filename directly in the program if not specifed, config 
            file will be parsed via argparse
        default: default config parser argument
    """
    def __init__(self, filename=None, default='config.yaml'):
        if filename is None:
            parser = argparse.ArgumentParser()
            parser.add_argument('--config', '-cfg', type=str, default=default)
            _cfg_name = parser.parse_args().config
        else:
            _cfg_name = filename

        assert type(_cfg_name) == str, 'invalid config filename specified (not a string type)'
        assert _cfg_name.rsplit('.', 1)[-1] == 'yaml', 'unspported config file type yet (not .yaml)'
        with open(_cfg_name, 'r') as _cfg_f:
            _raw_cfg = yaml.load(_cfg_f)
        
        self.config_path = _cfg_name
        self.config = _DotConfig(_raw_cfg)

        
    def get_config(self):
        return self.config
    
    def get_loss_func(self, attr):
        return getattr(F, attr)
    
    def get_optimizer_func(self, attr):
        return getattr(optim, attr)
    
    