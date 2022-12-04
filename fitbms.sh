#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("exp1data")
declare models=("MOS" "FLR" "RP") 

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method='bms'
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m='bms' 
        #python m2_simulate.py -d=$data_set -n=$model -f=1 -c=100 -m='bms'
    done  
done

## step 2: parameter recovery
declare data_sets=("param_recovery-MOS_fix")
declare models=("MOS_fix") 

for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method='bms'
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m='bms' 
        #python m2_simulate.py -d=$data_set -n=$model -f=1 -c=100 -m='bms' -r=0
    done  
done

## step 3: model recovery
declare data_sets=("exp1data-MOS_fix")
declare models=("MOS_fix" "FLR_fix" "RP_fix" "MOS" "FLR" "RP") 

for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method='bms'
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m='bms' 
        python m2_simulate.py -d=$data_set -n=$model -f=1 -c=100 -m='bms' -r=0
    done  
done


## step 2: visualize
#python m4_visualize.py 

