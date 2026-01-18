from util import get_project_path, join_path

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
import re
import fileinput
import atexit
import shutil
import threading

ROOT_PATH = get_project_path("")

# Kill all subprocesses on Ctrl-C exit

lock = threading.Lock()
processes = set()

def __kill():
    with lock:
        for p in processes:
            p.kill()

atexit.register(__kill)

def __train(batch: int, seq_len: int) -> int:
    p = subprocess.Popen(["sh", join_path(ROOT_PATH, "src/train.sh"), str(batch), str(seq_len)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with lock:
        processes.add(p)
    code = p.wait()
    with lock:
        processes.remove(p)
    return code

# TODO Test w/ slurm library
def generate_training_data(batches: int, seq_len: int, bif_max: int = 5):
    print(ROOT_PATH)
    if not os.path.exists(join_path(ROOT_PATH, "env/")):
        raise RuntimeError("Training environment has not been setup! Please run ./setup.sh or ./source.sh")
    
    # Remove previous build
    shutil.rmtree(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/training_data/output/"), ignore_errors=True)
    
    # Change the bif_max in the file because it is hardcoded
    with fileinput.input(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/training_data/run_single_batch.sh"), inplace=True) as lines:
        for line in lines:
            print(re.sub(r'^bif_max=[0-9]+', f"bif_max={bif_max}", line.rstrip())) # Printing outputs to the file
    
    print(f"Creating {batches} batches... (output hidden)")
    with ProcessPoolExecutor() as p:
        all(output == 0 for output in p.map(__train, range(1,1+batches), [seq_len] * batches))
        
    os.chdir(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/training_data/"))
    assert subprocess.call(["sh", "combine_batches.sh", str(batches), str(seq_len)]) == 0

# Only generates one model. The Bury paper uses the average given by 20 models in its results.
def generate_model(seq_len: int):
    
    lib_size = len(os.listdir(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/training_data/output/combined/output_resids/")))
    
    # Change the seq_len in the file because it is hardcoded
    with fileinput.input(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/dl_train/DL_training.py"), inplace=True) as lines:
        for line in lines:
            print(re.sub(r'^\(lib_size, ts_len\) = \([0-9]+, [0-9]+\)$', f"(lib_size, ts_len) = ({lib_size}, {seq_len})", line.rstrip())) # Printing outputs to the file
    
    os.chdir(join_path(ROOT_PATH, "env/deep-early-warnings-pnas/dl_train/"))
    subprocess.call(["python", "DL_training.py", "1", "1"])

generate_training_data(10, 100)
generate_model(100)
