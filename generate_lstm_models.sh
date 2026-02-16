#!/usr/bin/bash

OLD_DIR=$(pwd)
function on_exit() {
    cd $OLD_DIR
    exit 1
}
trap on_exit INT

DIR=$(dirname ${BASH_SOURCE[0]})

BATCHES=25

source $DIR/env.sh

for len in 500 1500 ;
do
    echo Training 20 models of length $len on $((4000*$BATCHES)) time series...
    for n in $(seq 1 10);
    do
        for pad in lrpad lpad ;
        do
            python -m models.lstm.train env/training/len$len/ -o output/models/lstm/len$len/ -n best_model_$n --pad $pad --train $len $BATCHES
        done
    done
done

cd $OLD_DIR
echo Finished!