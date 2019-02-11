import math


def get_thres_func(eps_start=1, 
                   eps_end=0, 
                   eps_decay=None, 
                   total_num=None, 
                   eps=1e-8,):
    
    if not total_num and not eps_decay:
        raise ValueError('must specify either eps_decay or total_num')
    if total_num and eps_decay:
        print('warning: both eps_decay and total_num specified, using total_num', flush=True)
    if total_num is not None:
        if total_num <= 0:
            raise ValueError('illegal total_num specified (should be > 0)')
        # using exponetial decay rate
        final_prop = (eps_end + eps) / eps_start
        def ret_func(curr_num):
            if curr_num > total_num:
                curr_num = total_num
            return eps_start * (final_prop ** (curr_num / total_num))
    else:
        if eps_decay < 0:
            raise ValueError('illegal eps_decay specified (should be >= 0)')
        def ret_func(curr_num):
            if curr_num > total_num:
                curr_num = total_num
            return eps_end + (eps_start - eps_end) * math.exp(
                    -curr_num / (eps_decay + eps))
    
    return ret_func


