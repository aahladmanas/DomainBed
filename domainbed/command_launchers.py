# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

def hpc_launcher(commands):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
    with open('jobs_{}.txt'.format(date_time), 'w') as f:
        f.write('\n'.join(commands) + '\n')
    num_tasks = len(commands)
    print("num_tasks", num_tasks)
    with open('submit_{}.sh'.format(date_time), 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        # f.write("#SBATCH --mem-per-cpu=1GB\n")
        f.write("#SBATCH --partition=p40_4,p100_4,v100_pci_2\n")
        f.write(f"#SBATCH --array=1-{num_tasks}\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --mem=32GB\n")
        f.write("#SBATCH --time=5-12:00:00\n")
        f.write("srun $(head -n $SLURM_ARRAY_TASK_ID jobs_{}.txt | tail -n 1)".format(date_time))
    subprocess.call("sbatch -vv submit_{}.sh".format(date_time), shell=True)

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'hpc': hpc_launcher,
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
