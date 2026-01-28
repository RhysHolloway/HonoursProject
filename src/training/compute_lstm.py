#!/usr/bin/python3

#SBATCH --job-name=batch
#SBATCH --partition=gpu-l4-n3
#SBATCH --qos=gpu-l4-n3
#SBATCH --cpus-per-task 11
#SBATCH --mem 40G

import os
import os.path
from typing import Final

# # Activate python venv
# ACTIVATE: Final = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))), "create-env.sh")
# print(ACTIVATE)
# os.system(f'/usr/bin/bash --rcfile {ACTIVATE}')

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import train_lstm
from cfut import SlurmExecutor
from concurrent.futures import Executor, ThreadPoolExecutor

MAX_CONCURRENT_SIMS: Final[int] = 1

# BATCHES: int = os.environ['SLURM_CPUS_PER_TASK']
# if BATCHES > 1:
#     BATCHES -= 1
# if BATCHES <= 0:
#     raise ValueError("Did not allocate enough cpus/tasks to batch the training data!")

import argparse
parser = argparse.ArgumentParser(
                prog='LSTM Training Data Generator',
                description='Generates training data')
parser.add_argument('output', type=str)
parser.add_argument('-b', '--batches', type=int)
parser.add_argument('-l', '--length', type=int)
parser.add_argument('-m', '--bifurcations', type=int, default=1000)
args = parser.parse_args()

BATCHES = args.batches

def sim_pool() -> Executor:
    return ThreadPoolExecutor(max_workers=10)

train_lstm.multibatch(
    path=args.output,
    batch_pool=SlurmExecutor(additional_setup_lines=f"""
        #SBATCH --job-name=simulate
        #SBATCH --partition=gpu-l4-n3
        #SBATCH --qos=gpu-l4-n3
        #SBATCH --cpus-per-task 11
        #SBATCH --mem 40G
    """),
    sim_pool=sim_pool,
    batches=BATCHES,
    ts_len=args.length,
    bif_max=args.bifurcations,
    max_concurrent_sims=MAX_CONCURRENT_SIMS,
)