#!/usr/bin/bash
DIR=$(dirname ${BASH_SOURCE[0]})

# Environment setup

ENV_DIR=$DIR/env

if [ ! -d $ENV_DIR ]; then
    echo "Creating environment directory..."
    mkdir $ENV_DIR
    ret=$?
    if [ $ret -ne 0 ]; then
        echo "Could not create environment directory in project root!"
        return
    fi
fi

# Python setup

PY_VENV_DIR=venv

if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    VENV_SCRIPT=bin/activate
else
    VENV_SCRIPT=Scripts/activate
fi

cd $ENV_DIR

if [ ! -d "$PY_VENV_DIR/" ]; then
    echo "Creating Python virual environment..."
    python -m venv $PY_VENV_DIR
    source "$PY_VENV_DIR/$VENV_SCRIPT"
    echo "Installing required packages..."
    pip install -r "../requirements.txt"
else
    echo "Activating virtual environment..."
    source "$PY_VENV_DIR/$VENV_SCRIPT"
fi

cd $DIR
