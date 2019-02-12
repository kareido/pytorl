import random
from collections import deque, namedtuple

class _ExpReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        if self.capacity is None:
            print('MEMORY WARNING: '
                  'capacity of exp replay not specified (infinity)', flush=True)
        elif self.capacity <= 0:
            raise ValueError('invalid capacity of replay queue specified')
        self.memory = deque(maxlen=self.capacity)
        
    def push(self, state):
        raise NotImplementedError
        
    def sample(self, batch_size=1):
        raise NotImplementedError
        
    def clear(self):
        self.memory.clear()
        
    def __len__(self):
        return len(self.memory)
        

def _get_namedtuple(obj_type):
    assert obj_type in {'std_DQN'}
    if obj_type == 'std_DQN':
        return namedtuple('std_DQN', 
                        ('curr_state', 'action', 'next_state', 'reward'))



class NaiveReplay(_ExpReplay):
    def __init__(self, capacity=None, obj_format='std_DQN'):
        super(NaiveReplay, self).__init__(capacity)
        self.obj_format = obj_format
        self.obj_type = _get_namedtuple(self.obj_format)
        
    def push(self, obj):
        self.memory.append(obj)
    
    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)
    
    def form_obj(self, *args):
        return self.obj_type(*args)