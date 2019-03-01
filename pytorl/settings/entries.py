import argparse
import os
import shutil
import socket
import subprocess
from pytorl.utils import ConfigReader

MODULE_NAME = 'pytorl'


def _cd_and_execute(trg_dir, command, run_name):
    os.chdir(str(trg_dir))
    env = os.environ.copy()
    env['run_name'] = run_name
    process = subprocess.Popen(command, shell = True, env = env)
    while True:
        try:
            process.wait()
            break
        except KeyboardInterrupt:
            print('\tPlease double press Ctrl-C within 1 second to kill srun job. '
                  'It will take several seconds to shutdown ...', flush = True)
        
        
def _get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip



def rl_run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', '-rn', default = 'default')
    parser.add_argument('--command', '-c', required = True)
    opt = parser.parse_args()

    setting_file = os.path.join(os.path.dirname(), '%s.yaml' % MODULE_NAME)
    cfg_reader = ConfigReader(default=setting_file)
    config = cfg_reader.get_config()
    exp_dir = config.experiment_dir
    
    src_dir = os.path.split(os.getcwd())[-1]
    if not os.path.isdir(exp_dir): 
        raise NotADirectoryError('experiment_dir [%s] does not exist' % exp_dir)

    exp_entry = os.join.path(exp_dir, opt.run_name)
    
    if os.path.isdir(exp_entry):
        while True:
            print('experiment [%s] already exists at [%s]:' % (opt.run_name, exp_dir), 
                  '\n>>>>>>>>>>>> overwrite it or not ? <<<<<<<<<<<< [Y/n]:', flush=True, end='')
            response = input().strip()
            if response in {'Y', 'y'}: break
            elif response in {'N', 'n'}: sys.exit()
            else: continue
        # warning: this overwrites previous experiment        
        shutil.rmtree(exp_entry)

    exp_entry.mkdir(parents = True, exist_ok = True)
    runway_file = runway_dir / '.pg-runway'
    trg_dir = os.path.join(exp_entry, src_dir)
    shutil.copytree(str(src_dir), str(trg_dir))
    
    _cd_and_execute(trg_dir, opt.command, opt.run_name)
    
    
def lrun(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntasks', '-n', type=int, required=True)
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    opt = parser.parse_args()
    cmd = ' '.join(opt.cmd)

    proc_list = []
    for proc_id in range(opt.ntasks):
        env = os.environ.copy()
        env['SLURM_NTASKS'] = str(opt.ntasks)
        env['SLURM_PROCID'] = str(proc_id)
        env['SLURM_NODELIST'] = '5412306 ' + _get_host_ip().replace('.', '-')
        proc_list.append(subprocess.Popen(cmd, shell=True, env=env))

    while True:
        try:
            for proc in proc_list:
                proc.wait()
            break
        except KeyboardInterrupt:
            print('\tPlease double press Ctrl-C within 1 second to kill srun job. '
                  'It will take several seconds to shutdown ...', flush=True)
