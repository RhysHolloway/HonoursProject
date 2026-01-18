Data tables must be placed under `/data` in order to be read and output plots are produced under `/output/plots/`

The code to build the models for the CNN-LSTM is in `src/build-model.py` and can be run with Python 3. You must run `source setup.sh` before running the model as it requires the libraries to create the training data to be compiled.

The functions to run the models are placed under `/src/main.py`, which can be run after installing the packages listed in `/install-packages.sh`