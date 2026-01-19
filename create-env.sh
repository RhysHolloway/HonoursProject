#!/bin/bash
DIR=$(dirname ${BASH_SOURCE[0]})
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    VENV_SCRIPT=bin/activate
else
    VENV_SCRIPT=Scripts/activate
fi
if [ ! -d "$DIR/env/" ]; then

    CUR_DIR="$(pwd)"
    mkdir "$DIR/env/"
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "Could not create env directory!"
        return
    fi
    cd "$DIR/env/" 

    echo "Creating Python virual environment..."
    python -m venv bury-venv
    source bury-venv/$VENV_SCRIPT

    git clone https://github.com/ThomasMBury/deep-early-warnings-pnas/
    cd deep-early-warnings-pnas/
    git checkout d56b5df879efbd557dad38aca1281cd00dc404c9 > /dev/null

    sed -i -e 's/random.seed(datetime.now())/random.seed(datetime.now().timestamp())/g' dl_train/DL_training.py # Fix a Python error
    sed -i -e 's=training_data/output_full=training_data/output=g' dl_train/DL_training.py # Fix trying to open a missing directory
    sed -i -e 's/ewstools>=2.1.2,<3.0/ewstools==2.0.2/g' requirements.txt # Required as a function used in the Bury 2021 paper was removed from the repo and the code never updated to fix it
    sed -i -e 's/python3/python/g' training_data/run_single_batch.sh training_data/combine_batches.sh # Fix if missing python3 command
    python ../scripts/fix_models.py dl_train/best_models_tf215/len500/
    python ../scripts/fix_models.py dl_train/best_models_tf215/len1500/

    
    cd ..

    pip install ruptures -r deep-early-warnings-pnas/requirements.txt -r ../requirements.txt

    git clone https://github.com/auto-07p/auto-07p/ auto
    cd auto
    git checkout 44cc1c42a888179bdbd608a40613caee9cbc4675 > /dev/null
    sh ./configure
    make
    python -m pip install ./
    source cmds/auto.env.sh

    cd $CUR_DIR

else # `deactivate` to leave python venv
    echo "Activating virtual environment..."
    source "$DIR/env/bury-venv/$VENV_SCRIPT"
    source "$DIR/env/auto/cmds/auto.env.sh"
fi
