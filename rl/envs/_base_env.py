

"""
this module provides base class/metaclass for creating rl environment using this base can help 
make your environment compatible with the exsiting implementation of rl algorithms since it 
regulates and implements some interfaces and standard functionalities, which are:
    I. public interfaces:
        1) a reset method that supports init the first frame
        2) a sample method that samples a random action
        3) a step method that takes an action and returns observation, 
            reward, done, info
        4) an action space getter
        5) a current observation getter
        
    II. common functions and statistics:
        1) a global frame counter and resetter method
        2) a current episodic frames counter and resetter method 
        3) a past episodes counter and resetter (i.e. resets counter)
        4) a frame stack setter that specifies how many frames that are stacked as an 
            stacked observation
        5) a frame-per-ation setter that specifies how many frames that current action 
            should repeat
        6) a single life mode setter
        7) an episodic initialization mode setter
        8) a tensorboard setter
        9) an episodic and per-action reward resetter and getter
       10) a global reinit method

if you create your own environment from scratch or you import an env from other sources and that 
env does not require a bass class, you must use base class _Env as your base class

if you retrieve your learning environment from other sources and taht env requires a base class 
other than _Env (e.g. Open AI gym envrionments usually requires a wrapper as their base class), 
in that case, you must use base metaclass _MetaEnv as your base class and make your environment 
class code looks like this example:
    class AtariWrapper(gym.Wrapper, metaclass=_MetaEnv):
        def __init__(self, env):
            blahblahblah...
        other stuff...
"""


class Env(object):
    """
    base class for rl environment
    """
    def __init__(self):
        self._action_reward = None
        self._episodic_frames = 0
        self._episodic_reward = None
        self._frames_action = 1
        self._frames_stack = 1
        self._global_episodes = 0
        self._global_frames = 0
        self._single_life = False
        self._tensorboard = None
    
    def current_ob(self):
        raise NotImplementedError()
    
    def num_actions(self):
        raise NotImplementedError()
    
    def sample(self):
        raise NotImplementedError()
        
    def reset(self):
        raise NotImplementedError()
        
    def step(self, action):
        raise NotImplementedError()
        
        
    def action_reward(self, num=None):
        """
        [!]WARNING: should check the legitimacy of num by yourself
        """
        if num is not None:
            self._action_reward = num
        return self._action_reward
    
    
    def episodic_frames(self, pattern=None, adder=1):
        if pattern == 'add':
            assert type(adder) == int and adder >= 0
            self._episodic_frames += adder
        if type(pattern) == int:
            assert pattern >= 0
            self._episodic_frames = pattern           
        return self._episodic_frames
    
    
    def episodic_reward(self, pattern=None, adder=None):
        if pattern == 'add':
            """
            [!]WARNING: should check the legitimacy of adder by yourself
            """
            assert adder is not None
            self._episodic_reward += adder
        elif pattern is not None:
            self._episodic_reward = pattern
        return self._episodic_reward
    
    
    def frame_action(self, pattern=None, num=None):
        if pattern == 'set':
            assert type(num) == int and num >= 1
            self._frames_action = num
        return self._frames_action
    
    
    def frames_stack(self, pattern=None, num=None):
        if pattern == 'set':
            assert type(num) == int and num >= 1
            self._frames_stack = num
        return self._frames_stack
    
    
    def global_frames(self, pattern=None, adder=1):
        if pattern == 'reset':
            self._global_frames = 0
        if pattern == 'add':
            assert type(adder) == int and adder >= 0
            self._global_frames += adder
        if type(pattern) == int:
            assert pattern >= 0
            self._global_frames = pattern           
        return self._global_frames
    
    
    def global_episodes(self, pattern=None, adder=1):
        if pattern == 'add':
            assert type(adder) == int and adder >= 0
            self._global_episodes += adder
        if type(pattern) == int:
            assert pattern >= 0
            self._global_episodes = pattern           
        return self._global_episodes 
    
    
    def single_life(self, flag=None):
        if flag is not None:
            self._single_life = flag
        return self._single_life
    
    
    def tensorboard(self, obj=None):
        """
        [!]WARNING: should check the legitimacy of adder by yourself
        """
        if obj is not None:
            self._tensorboard = obj
        return self._tensorboard
        
        
    def reset_statistics(self, mode):
        assert mode in {'all', 'episodic'}
        self._action_reward = None
        self._episodic_frames = 0
        self._episodic_reward = None
        if mode == 'all':
            self._global_episodes = 0
            self._global_frames = 0
       
    
    def reset_all(self):
        self.__init__()
        

        
"""
these methods help the setup process of metaclass: MetaEnv
"""        
def _get_attrs_setter(target):
    def _attrs_setter(attrs, values):
        if hasattr(attrs, '__iter__'):
            assert len(attrs) == len(values)
            for attr, val in zip(attrs, values):
                assert type(attr) == str
                if not hasattr(target, attr):
                    setattr(target, attr, val)
        else:
            if not hasattr(target, attrs):
                setattr(target, attrs, values)            
    return _attrs_setter

def _get_attr_setter(target):
    def _attr_setter(attr, value):
        if not hasattr(target, attr):
            setattr(target, attr, value)
    return _attr_setter
        
    
class MetaEnv(type):
    """
    base metaclass for third-party rl environment
    
    bases[0] is supposed to be the direct base class for the env and this metaclass will form the 
    part which the base class of the env does not cover, and will keep other settings for base 
    class to decide, if base class has the same attributes as the this metaclass has, base class 
    will OVERRIDE metaclass attribution under these circumstances
    
    """
    def __new__(self, name, bases, fields):
        instance = super(MetaEnv, self).__new__(self, name, bases, fields)
        """
        [!]WARNING: should check if this condition(direct_base = bases[0]) holds 
        """
        direct_base = bases[0]
        attr_setter = _get_attr_setter(direct_base)
        # get base 
        base_env = Env()
        # set base value attributions
        base_attrs, base_vals = zip(*base_env.__dict__.items())
        # have to wrap map as an Iterable to help the map func to work
        tuple(map(attr_setter, base_attrs, base_vals))
        # set base functionalities
        base_func_names = (attr for attr in dir(base_env) if not attr.startswith('_'))
        base_func_vals = (getattr(base_env, attr) for attr in base_func_names)
        # have to wrap map as an Iterable to help the map func to work
        tuple(map(attr_setter, base_func_names, base_func_vals))
        
        return instance
    
    
    