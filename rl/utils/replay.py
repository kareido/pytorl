import random
from collections import deque


class _ExpReplay:
    def __init__(self, zipper, capacity=None, sync=True):
        self.capacity, self.sync = capacity, sync
        if self.capacity is None:
            print('MEMORY WARNING: capacity of exp replay not specified (infinity)', flush=True)
        elif self.capacity <= 0:
            raise ValueError('invalid capacity of replay queue specified')
        self.memory = deque(maxlen=self.capacity)
        self.zipper = zipper
        
    def push(self, state):
        raise NotImplementedError
        
    def sample(self, batch_size=1):
        raise NotImplementedError
        
    def clear(self):
        self.memory.clear()
        
    def __len__(self):
        return len(self.memory)
        

class _Non_Sync_Replay(_ExpReplay):
    def __init__(self, **kwargs):
        super(_Non_Sync_Replay, self).__init__(**kwargs)
        
    def push(self, state):
        self.memory.append(state)
    
    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)
    
    
class _Sync_Replay(_ExpReplay):
    def __init__(self, **kwargs):
        super(_Sync_Replay, self).__init__(**kwargs)
        raise NotImplementedError('currently not available')


        
def get_exp_replay(zipper, capacity=None, sync=True):
    kwargs = {
        'zipper': zipper, 
        'capacity': capacity,
        'sync': sync,
    }
    if sync:
        return _Sync_Replay(**kwargs)
    
    return _Non_Sync_Replay(**kwargs)
        
        
        
        
        
        
        
        
        
        
        
