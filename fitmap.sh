#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

## declare all models and all data sets
declare data_sets=("gain_exp1data" "loss_exp1data")
declare models=("GagModel") #"MixPol" "GagModel" "RlRisk"

## step 1: fit the model
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method='map'
        #python m1_fit.py -d=$data_set -n=$model -s=420 -f=40 -c=40 -m='mle' 
        python m2_simulate.py -d=$data_set -n=$model -f=10 -c=10 -m='map'
    done  
done

## step 2: visualize
python m3_visualize.py 

