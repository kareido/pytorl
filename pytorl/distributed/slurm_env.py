import os


def get_world_size():
    return int(os.environ['SLURM_NTASKS'])


def get_rank():
    return int(os.environ['SLURM_PROCID'])


def get_jobid():
    return int(os.environ['SLURM_JOBID'])


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)