import os
from tensorboardX import SummaryWriter


"""
SummaryWriter wrapper
"""
class _TensorboardWriter(SummaryWriter):
    def __init__(self, logdir):
        super(_TensorboardWriter, self).__init__(logdir)
        self.logdir = logdir
    
    def add_textfile(self, tag, path, filename=None):
        assert type(tag) == str
        if filename is not None:
            path = os.path.join(path, filename)
        with open(path, 'r') as f:
            # since 'list' object has no attribute 'encode'
            content = '\n'.join(f.readlines())
        self.add_text(tag, content)
        

def get_tensorboard_writer(logdir):
    writer = _TensorboardWriter(logdir)
    
    return writer


