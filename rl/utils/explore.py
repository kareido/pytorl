import math


def get_epsilon_greedy_func(eps_start, 
                            eps_end,  
                            steps, 
                            decay='linear', 
                            eps=1e-8,):
    assert decay in {'exponential', 'linear'}
    if steps <= 0:
        raise ValueError('illegal steps specified (should be > 0)')
    if decay == 'exponential':
        # using exponetial decay rate
        final_prop = (eps_end + eps) / eps_start
        def ret_func(curr_num):
            if curr_num > steps:
                curr_num = steps
            return eps_start * (final_prop ** (curr_num / steps))
        
    else:
        # using linear decay rate
        decay_rate = (eps_start - eps_end) / (steps + eps)
        def ret_func(curr_num):
            if curr_num > steps:
                curr_num = steps
            return eps_start - curr_num * decay_rate
    
    return ret_func


