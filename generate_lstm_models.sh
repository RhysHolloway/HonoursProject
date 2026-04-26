#!/usr/bin/bash

OLD_DIR=$(pwd)
function on_exit() {
    cd $OLD_DIR
    exit 1
}
trap on_exit INT

DIR=$(dirname ${BASH_SOURCE[0]})

BATCHES_500=${BATCHES_500:-125}
BATCHES_1500=${BATCHES_1500:-50}
JOBS=${JOBS:-${SLURM_CPUS_PER_TASK:--1}}

cd $DIR
git fetch origin && git pull
source setup.sh

for len in 500 1500 ;
do
    case "$len" in
        500) BATCHES=$BATCHES_500 ;;
        1500) BATCHES=$BATCHES_1500 ;;
    esac
    echo Training 20 models of length $len on $((4000*$BATCHES)) time series...
    for n in $(seq 1 10);
    do
        for pad in lrpad lpad ;
        do
            python -m src.lstm.train env/training/len$len/ -o output/models/lstm/len$len/ -n best_model_$n --pad $pad --train $len $BATCHES --jobs "$JOBS"
        done
    done
done

cd $OLD_DIR
echo Finished!
