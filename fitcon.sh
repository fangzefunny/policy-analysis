#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("for_interpret_HC" "for_interpret_PAT")
declare models=("RS_test")  

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method='bms'
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m='bms' #2>&1 | tee output.txt
    done  
done


