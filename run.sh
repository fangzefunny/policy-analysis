#!/bin/bash

declare data_sets=("exp1data")
declare method='map'
declare alg='BFGS'

## STEP 1: Preprocess the data 
python m0_preprocess.py

## STEP 2: Fit models to the data and estimate the parameters 
declare models=("MOS6" "MOS22" "FLR6" "FLR22" "RS3" "RS13" "PH4" "PH17" 
                "EU_MO" "EU_HA" "MO_HA" "PS_MO_HA" "EU_RD_HA" 
                "EU_PS_MO_HA" "EU_MO_HA_RD" "linear_comb")
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method=$method
        python m1_fit.py -d=$data_set -n=$model -s=420 -f=30 -c=30 -m=$method -a=$alg
        python m2_rollout.py -d=$data_set -n=$model -f=20 -c=20 -m=$method -a=$alg
    done  
done

## STEP 3: Simulate for interpretation
python m4_simulations  

## STEP 4: Model and parameter recovery 
declare models=("MOS6" "MOS22" "FLR6" "FLR22" "RS3" "RS13" "PH4" "PH17") 
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method=$method
        python m3_recover.py -d=$data_set -n=$model -s=420 -c=40 -m=$method -a=$alg
    done  
done

