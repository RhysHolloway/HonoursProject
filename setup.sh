#!/usr/bin/env bash
CUR_DIR="$(cd)"
cd "$(dirname "$0")"

mkdir env
cd env

if [ ! -d "deep-early-warnings-pnas" ]; then
    git clone https://github.com/ThomasMBury/deep-early-warnings-pnas/
    cd deep-early-warnings-pnas/
    git checkout d56b5df879efbd557dad38aca1281cd00dc404c9 > /dev/null
    sed -i -e 's/random.seed(datetime.now())/random.seed(datetime.now().timestamp())/g' dl_train/DL_training.py # Fix a Python error
    sed -i -e 's=training_data/output_full=training_data/output=g' dl_train/DL_training.py # Fix trying to open a missing directory
    sed -i -e 's/ewstools>=2.1.2,<3.0/ewstools==2.0.2/g' requirements.txt # Required as a function used in the Bury 2021 paper was removed from the repo and the code never updated to fix it
    cd ..
fi

if [ ! -d "bury-venv" ]; then
    echo "Creating Python venv..."
    python -m venv bury-venv
    source bury-venv/bin/activate
    pip install ruptures -r deep-early-warnings-pnas/requirements.txt
else
    source bury-venv/bin/activate
fi

if [ ! -d "auto" ]; then
    git clone https://github.com/auto-07p/auto-07p/ auto
    cd auto
    git checkout 44cc1c42a888179bdbd608a40613caee9cbc4675 > /dev/null
    sh ./configure
    make
    pip install ./
fi

deactivate
cd $CUR_DIR