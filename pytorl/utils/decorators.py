import functools
from collections import defaultdict

__all__ = ['Setting']


class DecorateSetting:
    def __init__(self):
        self.instances = defaultdict(set)

        
    def __call__(self, func):
        @functools.wraps(func)
        def _setting(caller, *args, **kwargs):
            if func.__name__ in self.instances[hash(caller)]: 
                print('warning: %s of %s has been overridden' % (func.__name__, caller), flush=True)
            self.instances[hash(caller)].add(func.__name__)
            func(caller, *args, **kwargs)
        return _setting

    
    def only_once(self, func):
        @functools.wraps(func)
        def _check_setting(caller, *args, **kwargs):
            if func.__name__ in self.instances[hash(caller)]: 
                raise RuntimeError('%s can only be called once' % func)
            self.instances[hash(caller)].add(func.__name__)
            func(caller, *args, **kwargs)
        return _check_setting
    
    
Setting = DecorateSetting()


