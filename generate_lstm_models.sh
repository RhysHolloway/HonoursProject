#!/usr/bin/bash

OLD_DIR=$(pwd)
function on_exit() {
    cd $OLD_DIR
    exit 1
}
trap on_exit INT

DIR=$(dirname ${BASH_SOURCE[0]})

BATCHES=10

source $DIR/env.sh

for len in 500 1500 ;
do
    echo Generating training data of length $len...
    python training/generate.py env/training/len$len/ -b $BATCHES -l $len
    echo Training $((2*$BATCHES)) models of length $len...
    for n in $(seq 1 $BATCHES);
    do
        for type in lrpad lpad ;
        do
            python training/train.py env/training/len$len/ -o models/lstm/len$len/ -n best_model_$n --type $type
        done
    done
done

echo Finished!
on_exit()