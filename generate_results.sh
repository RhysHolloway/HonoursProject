#!/usr/bin/bash

OLD_DIR=$(pwd)
function on_exit() {
    cd $OLD_DIR
    exit 1
}
trap on_exit INT

DIR=$(dirname ${BASH_SOURCE[0]})

cd $DIR
git fetch origin && git pull
source setup.sh
python -m models.run

cd $OLD_DIR