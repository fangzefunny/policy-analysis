#!/bin/bash

## step 0: preprocess the data 
python m0_preprocess.py

declare method='map'
declare alg='BFGS'

## declare all models and all data sets
declare data_sets=("exp1data")
#declare models=("MOS6" "MOS22" "FLR6" "FL22" "RS3" "RS13" "PH4" "PH17") 
# "MOS6" "MOS22" "FLR6" "FLR22" "RS3" "RS13" "PH4" "PH17" 
# "EU_MO" "EU_HA" "MO_HA" "PS_MO_HA" "EU_PS_MO_HA" EU_MO_HA_RD
# "EU_MO18" "EU_HA18" "MO_HA18" "PS_MO_HA22" "EU_PS_MO_HA26" "linear_comb"
declare method='map'
declare alg='BFGS'

# for data_set in "${data_sets[@]}"; do 
#     for model in "${models[@]}"; do 
#         echo Data set=$data_set  Model=$model Method=$method
#         python m1_fit.py -d=$data_set -n=$model -s=420 -f=30 -c=30 -m=$method -a=$alg
#         python m2_simulate.py -d=$data_set -n=$model -f=20 -c=20 -m=$method -a=$alg
#     done  
# done

# recover the model of interest
declare models=("FLR22") # "MOS6" "MOS22" "FLR6" "FLR22" "RS3" "RS13" "PH4" "PH17"
for data_set in "${data_sets[@]}"; do 
    for model in "${models[@]}"; do 
        echo Data set=$data_set  Model=$model Method=$method
        python m3_recover.py -d=$data_set -n=$model -s=420 -c=40 -m=$method -a=$alg
    done  
done

#python m2_simulate.py -d='exp1data-MOS6' -n='RS13' -f=20 -c=20 -m=$method -a=$alg
#python m2_simulate.py -d='exp1data-MOS22' -n='FLR19' -f=20 -c=20 -m=$method -a=$alg


