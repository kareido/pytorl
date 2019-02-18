import math


def eps_greedy_func(eps_start=1, eps_end=0.02, num_decays=1, 
                    global_frames_func=lambda: 1):
    """
    framed and linear decay of greedy threshold
    
    Args:
        counter_func: should be a  callable counter, someting like env.global_frames
    """
    assert num_decays >= 1 and type(num_decays) == int
    decay_rate = (eps_start - eps_end) / num_decays
    
    def _result():
        if global_frames_func() >= num_decays: return eps_end
        return eps_start - decay_rate * global_frames_func()
    
    return _result


""" [!]DEPRECATED
this method was too complicated so it got discarded, besides, I found nowhere to use exponential
epsilon greedy decay.
DO NOT USE IT
"""
def _deprecated_get_epsilon_greedy_func(eps_start, 
                            eps_end,  
                            steps, 
                            delay = 0, 
                            decay='linear', 
                            eps=1e-8,):
    assert decay in {'exponential', 'linear'}
    if steps <= 0:
        raise ValueError('illegal steps specified (should be > 0)')
    if decay == 'exponential':
        # using exponetial decay rate
        final_prop = (eps_end + eps) / eps_start
        def ret_func(curr_num):
            curr_num -= delay
            if curr_num < 0: curr_num = 0
            if curr_num > steps: curr_num = steps
            return eps_start * (final_prop ** (curr_num / steps))
        
    else:
        # using linear decay rate
        decay_rate = (eps_start - eps_end) / (steps + eps)
        def ret_func(curr_num):
            curr_num -= delay
            if curr_num < 0: curr_num = 0
            if curr_num > steps: curr_num = steps
            return eps_start - curr_num * decay_rate
    
    return ret_func