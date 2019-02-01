import argparse
import yaml


class DotConfig(dict):
    
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            return super().__getattr__(key)
        if isinstance(value, dict):
            return DotConfig(value)
        return value

    
def get_config(filename=None):
    if filename is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-cfg', type=str, default='config.yaml')
        _cfg_name = parser.parse_args()
    else:
        _cfg_name = filename
    
    assert type(_cfg_name) == str, 'invalid config filename specified (not a string type)'
    assert _cfg_name.rsplit('.', 1)[-1] == 'yaml', 'unspported config file type yet (not .yaml)'
    
    with open(_cfg_name, 'r') as _cfg_f:
        _raw_cfg = yaml.load(_cfg_f)
    
    return DotConfig(_raw_cfg)
