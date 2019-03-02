



def eps_greedy_func(eps_start=1, eps_end=0.02, num_decays=1, 
                    global_frames_func=lambda: 1):
    """
    framed and linear decay of greedy threshold
    
    Args:
        global_frames_func: should be a callable counter, someting like env.global_frames
    """
    assert num_decays >= 1 and type(num_decays) == int
    assert eps_start >= eps_end
    decay_rate = (eps_start - eps_end) / num_decays
    
    def _result():
        if global_frames_func() >= num_decays: return eps_end
        return eps_start - decay_rate * global_frames_func()
    
    return _result


def beta_priority_func(beta_start=0, beta_end=1, num_incres=1,
                       global_frames_func=lambda: 1):
    """
    framed and linear increase of the effect of importance weights in DQN's prioritized replay
    
    Args:
        global_frames_func: should be a callable counter, someting like env.global_frames
    """
    assert num_incres >= 1 and type(num_incres) == int
    assert beta_end >= beta_start
    incre_rate = (beta_end - beta_start) / num_incres
    
    def _result():
        if global_frames_func() >= num_incres: return beta_end
        return beta_start + incre_rate * global_frames_func()
    
    return _result