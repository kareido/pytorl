import random
from collections import deque, namedtuple
import torch
from ._tree import SumSegmentTree, MinSegmentTree


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
            
        
    def __len__(self):
        raise NotImplementedError
        
    def push(self, state):
        raise NotImplementedError
        
    def sample(self):
        raise NotImplementedError
        
    def clear(self):
        raise NotImplementedError

    

class VanillaReplay(_ExpReplay):
    def __init__(self, capacity=None, batch_size=32, init_size=None):
        super(VanillaReplay, self).__init__(capacity, batch_size, init_size)
        self.obj_type = namedtuple('Exp', ('curr_state', 'action', 'next_state', 'reward'))
        
        self.memory = deque(maxlen=self.capacity)
    
    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()
        
    def push(self, *obj):
        processed_obj = self.form_obj(*obj)
        self.memory.append(processed_obj)
    
    def sample(self):
        return random.sample(self.memory, self.batch_size)
    
    def form_obj(self, *args):
        return self.obj_type(*args)
    
    
    
class LazyReplay(_ExpReplay):
    """
    this replay splits the stacked frames and makes sure each frame is only saved once, so it is
    meant to be memory-efficient. However, the sampling speed will be compromised since that
    process is kind of complicated
    """
    def __init__(self, capacity=None, batch_size=32, init_size=None, frames_stack=None):
        super(LazyReplay, self).__init__(capacity, batch_size, init_size)
        assert frames_stack is not None
        self.obj_type = namedtuple('Exp', ('curr_state', 'action', 'next_state', 'reward'))
        self.frames_stack = frames_stack
        self._valid_idx = deque([])
        self._valid_flag = [False] * self.capacity
        self._done = True
        self.memory = [None] * self.capacity
        self._idx = 0

    
    def clear(self):
        self.memory = [None] * self.capacity
        self._idx = 0
        self._valid_idx.clear()
    
    
    def __len__(self):
        return len(self._valid_idx)
    
    
    def push(self, *obj):
        """
        preprocess obj sequence and save it to the memory, note that preprocessing varies due to 
        various obj format
        """
        curr_state, action, next_state, reward = obj
        if self._done:
            initial_frames = curr_state.squeeze().split(1)
            for f in initial_frames:
                if self.memory[self._idx] is not None and self._valid_flag[self._idx]:
                    self._valid_idx.popleft()
                    self._valid_flag[self._idx] = False
                self.memory[self._idx] = (f, None, None)
                self._idx = (self._idx + 1) % self.capacity
            self._done = False
        if next_state is None:
            self._done = True
            next_frame = None
        else:
            next_frame = next_state.squeeze().split(1)[-1]
        if self.memory[self._idx] is not None and self._valid_flag[self._idx]:
            self._valid_idx.popleft()
            self._valid_flag[self._idx] = False
        self.memory[self._idx] = (next_frame, action, reward)
        self._idx = (self._idx + 1) % self.capacity
        self._valid_flag[(self._idx - 5) % self.capacity] = True
        self._valid_idx.append((self._idx - 5) % self.capacity)
            
               
    def sample(self):
        ret_list = []
        frames_buffer = deque([], maxlen=self.frames_stack)
        indices = random.sample(self._valid_idx, self.batch_size)
        for idx in indices:
            for shift in range(self.frames_stack):
                frames_buffer.append(self.memory[(idx + shift) % self.capacity][0])
            curr_state = torch.cat(tuple(frames_buffer)).unsqueeze(0).clone()
            next_frame, action, reward = self.memory[(idx + self.frames_stack) % self.capacity]
            if next_frame is None:
                next_state = None
            else:
                frames_buffer.append(next_frame)
                next_state = torch.cat(tuple(frames_buffer)).unsqueeze(0).clone()
            ret_list.append((curr_state, action, next_state, reward))
        return tuple(ret_list)
            
        
    def form_obj(self, *args):
        return self.obj_type(*args)        
    


class PrioritizedReplay(_ExpReplay):
    def __init__(
        self, 
        capacity=None, 
        batch_size=32, 
        init_size=None, 
        alpha=1, 
        beta_func=lambda: 0, 
        eps=1e-6
    ):
        super(PrioritizedReplay, self).__init__(capacity, batch_size, init_size)
        assert alpha >= 0 and capacity is not None
        self._alpha = alpha
        self.in_obj_type = namedtuple('Exp', ('curr_state', 'action', 'next_state', 'reward'))
        self.out_obj_type = namedtuple('PriorExp', (
            'curr_state', 'action', 'next_state', 'reward', 'weight', 'index'))
        self.memory = []
        self._idx = 0
        self._sum_prior = SumSegmentTree(self.capacity)
        self._min_prior = MinSegmentTree(self.capacity)
        self._max_prior = 1
        self.get_beta = beta_func
        self.eps = eps
        
        
    def clear(self):
        self.memory = []
        self._idx = 0
        self._sum_prior = SumSegmentTree(self.capacity)
        self._min_prior = MinSegmentTree(self.capacity)
    
    
    def __len__(self):
        return len(self.memory)
    
    
    def push(self, *obj):
        processed_obj = self._form_input_obj(*obj)
        if len(self) < self.capacity:
            self.memory.append(processed_obj)
        else:
            self.memory[self._idx] = processed_obj
            
        priority = self._max_prior ** self._alpha
        self._min_prior[self._idx] = self._sum_prior[self._idx] = priority
        self._idx = (self._idx + 1) % self.capacity
    
    
    def _sample_indices(self):
        indices = []
        for _ in range(self.batch_size):
            # sum(a, b): a is inclusive whereas b is exclusive
            mass = random.random() * self._sum_prior.sum(0, len(self))
            idx = self._sum_prior.find_prefixsum_idx(mass)
            indices.append(idx)
        return indices
    
    
    def sample(self):
        beta = self.get_beta()
        indices = self._sample_indices()
        weights = []
        ret_list = []
        p_min = self._min_prior.min() / self._sum_prior.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in indices:
            p_sample = self._sum_prior[idx] / self._sum_prior.sum()
            weight = (p_sample * len(self)) ** (-beta) / max_weight
            ret_list.append(tuple(list(self.memory[idx]) + [weight, idx]))
        
        return ret_list
    
    
    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0 and 0 <= idx < len(self.memory)
            self._sum_prior[idx] = priority ** self._alpha
            self._min_prior[idx] = priority ** self._alpha
            self._max_prior = max(self._max_prior, priority)
    
    
    def _form_input_obj(self, *args):
        return self.in_obj_type(*args)
        
        
    def form_obj(self, *args):
        return self.out_obj_type(*args)

    
    