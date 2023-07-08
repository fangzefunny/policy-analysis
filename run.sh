#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("exp1data")
declare models=("MOS6" "MOS18" "FLR6" "FLR15" "RS3" "RS9")
declare method='map'
declare alg='BFGS'

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method=$method
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m=$method -a=$alg
        python m2_simulate.py -d=$data_set -n=$model -f=10 -c=10 -m=$method
    done  
done


