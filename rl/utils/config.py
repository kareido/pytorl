import argparse
import yaml


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

    
def get_config(filename=None, default='config.yaml'):
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
