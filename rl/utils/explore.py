import math


def get_thres_func(eps_start, 
                   eps_end,  
                   steps, 
                   eps=1e-8,):
    
    if steps <= 0:
        raise ValueError('illegal steps specified (should be > 0)')
    # using exponetial decay rate
    final_prop = (eps_end + eps) / eps_start
    def ret_func(curr_num):
        if curr_num > steps:
            curr_num = steps
        return eps_start * (final_prop ** (curr_num / steps))
    
    return ret_func


