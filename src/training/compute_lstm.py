#!/usr/bin/python3

#SBATCH --job-name=batch
#SBATCH --partition=gpu-l4-n3
#SBATCH --qos=gpu-l4-n3
#SBATCH --cpus-per-task 11
#SBATCH --mem 40G

import os
import os.path

# # Activate python venv
# ACTIVATE: Final = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "create-env.sh")
# print(ACTIVATE)
# os.system(f'/usr/bin/bash --rcfile {ACTIVATE}')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import train_lstm

from cfut import SlurmExecutor
from concurrent.futures import Executor, ThreadPoolExecutor

def default_batches():
    slurm_batches: int | None = int(os.environ.get('SLURM_CPUS_PER_TASK'))
    if slurm_batches is not None:
        if slurm_batches > 1:
            slurm_batches -= 1
    return slurm_batches

import argparse
parser = argparse.ArgumentParser(
                prog='LSTM Training Data Generator',
                description='Generates training data')
parser.add_argument('output', type=str)
parser.add_argument('-b', '--batches', type=int, default=default_batches())
parser.add_argument('-l', '--length', type=int)
parser.add_argument('-m', '--bifurcations', type=int, default=1000)
args = parser.parse_args()

def sim_pool() -> Executor:
    return ThreadPoolExecutor(max_workers=10)

train_lstm.multibatch(
    path=args.output,
    batch_pool=SlurmExecutor(additional_setup_lines=[
        "#SBATCH --job-name=simulate"
        "#SBATCH --partition=gpu-l4-n3"
        "#SBATCH --qos=gpu-l4-n3"
        "#SBATCH --cpus-per-task 11"
        "#SBATCH --mem 40G"
    ]),
    sim_pool=sim_pool,
    batches=args.batches,
    ts_len=args.length,
    bif_max=args.bifurcations,
    max_task_count=lambda _: int(os.environ.get(["SLURM_CPUS_PER_TASK"]) or 1)
)