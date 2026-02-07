# Sudden transitions in Earth's history

This repository contains code to reproduce data and results in Rhys Holloway's honours project.

## Setup

Before running code, a local environment for the project can be both activated with the Bash shell script `env.sh`, which when called with the command `source env.sh` activates the environment.

During the first run the script creates a virtual Python environment and downloads the required libraries.

## Reproduction

### Identifying tipping points through classification of bifurcations in a time-series by a deep learning model.

#### Generation of the training set

In order to produce results, the model needs to be trained, and that data needs to be generated.

The python script at `training/generate.py` is runnable and takes in a path argument defining the output directory for the generations, `-b` defining batch count (for training multiple models), `-l` for defining the time-series length the script will generate and models will train on, and optionally `-m` for defining the number of bifurcations to train on (defaults to n=1000, which turns into 4*n simulations consisting of n forced Hopf, Fold and Transcritical simulations each and a total of n null Hopf, Fold and Transcritical simulations).

The script can resume previous generation if it is stopped during a run.

The commands for generation used to achieve the results shown in the project are `python training/generate.py env/training/len500 -b 10 -l 500` and `python training/generate.py env/training/len1500 -b 10 -l 1500`.

#### Training the models

The python script at `training/train.py` can be run in order to train models. It takes in a path argument defining the directory of the training data generated from previous runs of the generator script. This input directory can read recursively for multiple runs of data, such as over multiple time-series lengths. It also takes in another path argument defining the output directory to place the completed deep learning models in.

The command for training the models used to achieve the results shown in the project are `python training/train.py env/training/len$TS_LEN/ -o models/lstm/len$TS_LEN/ -n best_model_$N` where TS_LEN is the time series length from the generated data and N is the nth model trained.

#### Running the models

The `LSTM` class can be used to run models on datasets. It requires a Keras model picking function to be passed in as well, which is provided in the form of the `LSTMModels` class which reads models from the folder structure generated using the above training instructions.

The code in `src/main.py` was used to generate results shown in the project.

### Metric-based analysis

### DLM