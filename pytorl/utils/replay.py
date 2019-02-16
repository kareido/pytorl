import random
from collections import deque, namedtuple


class _ExpReplay:
    def __init__(self, capacity, batch_size, init_size):
        assert batch_size is not None and batch_size >= 1
        if type(init_size) != int or init_size < batch_size: init_size = batch_size
        self.capacity = capacity
        self.batch_size = batch_size
        self.init_size = init_size
        if self.capacity is None:
            print('MEMORY WARNING: '
                  'capacity of exp replay not specified (infinity)', flush=True)
        elif self.capacity <= 0:
            raise ValueError('invalid capacity of replay queue specified')
            
        self.memory = deque(maxlen=self.capacity)
        
        
    def push(self, state):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
        
    def clear(self):
        self.memory.clear()
        
    def __len__(self):
        return len(self.memory)
        

def _get_namedtuple(obj_type):
    assert obj_type in {'std_DQN'}
    if obj_type == 'std_DQN':
        return namedtuple('std_DQN', ('curr_state', 'action', 'next_state', 'reward'))



class VanillaReplay(_ExpReplay):
    def __init__(self, obj_format, capacity=None, batch_size=32, init_size=None):
        super(VanillaReplay, self).__init__(capacity, batch_size, init_size)
        self.obj_format = obj_format
        self.obj_type = _get_namedtuple(self.obj_format)
        
    def push(self, obj):
        self.memory.append(obj)
    
    def sample(self):
        return random.sample(self.memory, self.batch_size)
    
    def form_obj(self, *args):
        return self.obj_type(*args)