import argparse 
import os 
import pickle
import datetime 
import numpy as np
import pandas as pd
import subprocess

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',    '-d', help='which_data', type = str, default='exp1data')
parser.add_argument('--env_name',    '-e', help='which environment', type = str, default='rl_reversal')
parser.add_argument('--method',      '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--algorithm',   '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--agent_name',  '-n', help='choose agent', default='MOS6')
parser.add_argument('--other_agent', '-o', help='choose agent', default=
                                "['MOS6', 'MOS22', 'FLR6', 'FLR22', 'RS3', 'RS13', 'PH4', 'PH17']")
parser.add_argument('--n_cores',     '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',        '-s', help='random seed', type=int, default=420)
args = parser.parse_args()
args.agent = eval(args.agent_name)
env =  eval(args.env_name)
args.group = 'group' if args.method=='hier' else 'ind'

pth = os.path.dirname(os.path.abspath(__file__))

# -------------------------------- #
#        PARAMETER RECOVERY        #
# -------------------------------- #

def syn_data_param_recover_paral(pool, data, args, n_samp=20):

    # set seed, conditions
    rng = np.random.RandomState(args.seed+1)
    n_param = args.agent.n_params

    # get mean parameters 
    fname  = f'{pth}/fits/{args.data_set}/avg_params'
    fname += f'-{args.agent_name}-{args.method}.csv'
    param  = pd.read_csv(fname, index_col=0).iloc[0, 0:n_param].values

    # create sythesize params for different conditions
    truth_params = {p: [] for p in args.agent.p_name}
    truth_params['sub_id'] = []

    # the index paramters of interest
    poi_id = [args.agent.p_name.index(p) for p in args.agent.p_poi]

    # get params for parameter recovery 
    bnds = args.agent.p_pbnds
    n_sub = len(poi_id)*n_samp
    for sub_id in range(n_sub):
        p_temp = param.copy()
        p_temp[0] = np.log(0.423)-np.log(1-0.423)
        p_temp[1] = np.log(10.803)
        for i in poi_id:
            p_temp[i] = bnds[i][0]+rng.rand()*(bnds[i][1]-bnds[i][0])
        for i in range(args.agent.n_params):
            truth_params[args.agent.p_name[i]].append(p_temp[i])
        truth_params['sub_id'].append(sub_id)
        
    ## save the ground turth parameters             
    truth_params_lst = pd.DataFrame.from_dict(truth_params)
    fname = f'{pth}/data/param_recover-truth-median-{args.data_set}-{args.agent_name}.csv'
    truth_params_lst.to_csv(fname)
    
    ## start simulate with the generated parameters  
    res = [pool.apply_async(syn_data_param_recover, args=[row, data, args.seed+2+2*i])
                            for i, row in truth_params_lst.iterrows()]
    data_for_recovery = {}
    sub_lst = truth_params_lst['sub_id']
    for i, p in enumerate(res):
        data_for_recovery[sub_lst[i]] = p.get() 
    
    fname = f'{pth}/data/param_recover-median-{args.agent_name}.pkl'
    with open(fname, 'wb')as handle:
        pickle.dump(data_for_recovery, handle)
    print(f'Synthesize data (param recovery) for {args.agent_name} has been saved!')

def syn_data_param_recover(row, data, seed, n_block=10):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(args.agent, env_fn=env)
    ind = rng.choice(list(data.keys()), size=n_block)
    param = list(row[args.agent.p_name].values)
    recovery_data = {}
    for idx in ind:
        block_data = data[idx][list(data[idx].keys())[0]]
        sim_data = model.sim({0: block_data}, param, rng)
        sim_data = sim_data.drop(columns=args.agent.voi)
        recovery_data[idx] = sim_data

    return recovery_data

def param_recover(args):

    ## STEP 0: GET PARALLEL POOL
    print(f'Parameter recovering {args.agent_name}...')
    pool = get_pool(args)

    ## STEP 1: SYTHESIZE FAKE DATA FOR PARAM RECOVER
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    syn_data_param_recover_paral(pool, data, args, n_samp=20)
    pool.close() 

    ## STEP 2: REFIT THE MODEL TO THE SYTHESIZE DATA 
    cmand = ['python', 'm1_fit.py', f'-d=param_recover-median-{args.agent_name}',
              f'-n={args.agent_name}', '-s=420', '-f=40',
              '-c=40', f'-m={args.method}', f'-a={args.algorithm}']
    subprocess.run(cmand)

# -------------------------------- #
#          MODEL RECOVERY          #
# -------------------------------- #

def syn_data_model_recover_paral(pool, data, args, n_samp=20):

    # set seed, conditions
    rng = np.random.RandomState(args.seed+2)
    n_param = args.agent.n_params

    # get parameters 
    fname  = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'      
    with open(fname, 'rb')as handle: fit_info_orig = pickle.load(handle)

    ## create a sub list of subject list 
    sub_lst_orig = list(fit_info_orig.keys())
    if 'group' in sub_lst_orig: sub_lst_orig.pop(sub_lst_orig.index('group'))
    # get sub for simulate
    sub_lst = ['n33', 'n24', 'cb1', 'cb45', 'cb79', 'cb17', 'cb80', 'cb63', 'cb7',
                'n31', 'n19', 'n8', 'cb17', 'n33', 'n36', 'cb45', 'cb68', 'n19',
                'cb7', 'cb79', 'cb13', 'cb46', 'cb49', 'cb20', 'cb84', 'cb74', 'cb14', 'cb200',
                'cb75', 'cb13', 'cb20', 'cb46', 'cb105', 'cb108', 'cb13', 'cb30',
                'cb74', 'cb49', 'cb96', 'cb14']
    #rng.choice(sub_lst_orig, size=n_samp)
    fit_param = {k: fit_info_orig[k]['param'] for k in sub_lst}

    # create the synthesize data for the chosen sub
    res = [pool.apply_async(syn_data_model_recover, 
                    args=(data, fit_param[sub_id], sub_id, args.seed+5*i))
                    for i, sub_id in enumerate(sub_lst)]

    syn_data = {}
    for _, p in enumerate(res):
        sub_id, sim_data = p.get() 
        syn_data[sub_id] = sim_data 

    # save for fit 
    with open(f'{pth}/data/{args.data_set}-{args.agent_name}.pkl', 'wb')as handle:
        pickle.dump(syn_data, handle)
    print(f'Synthesize data (model_recovery) for {args.agent_name} has been saved!')

def syn_data_model_recover(data, param, sub_id, seed, n_samp=10):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(args.agent, env_fn=env)

    # synthesize the data and save
    sim_data = {} 
    task_ind = rng.choice(list(data.keys()), size=n_samp)
    
    for i, task_id in enumerate(task_ind):
        block_id = rng.choice(list(data[task_id].keys()))
        task = data[task_id][block_id]
        sim_sample = model.sim({i: task}, param, rng=rng)
        sim_sample = sim_sample.drop(columns=model.agent.voi)
        sim_data[i] = sim_sample
     
    return sub_id, sim_data

def model_recover(args):

    ## STEP 0: GET PARALLEL POOL
    print(f'Model recovering {args.agent_name}...')
    pool = get_pool(args)

    ## STEP 1: SYTHESIZE FAKE DATA FOR PARAM RECOVER
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    syn_data_model_recover_paral(pool, data, args, n_samp=20)
    pool.close() 

    ## STEP 2: REFIT THE OTHER MODEL TO THE SYTHESIZE DATA 
    for agent_name in eval(args.other_agent):
        cmand = ['python', 'm1_fit.py', f'-d={args.data_set}-{args.agent_name}',
                f'-n={agent_name}', '-s=420', '-f=40',
                '-c=40', f'-m={args.method}', f'-a={args.algorithm}']
        subprocess.run(cmand)


if __name__ == '__main__':

    param_recover(args)

    model_recover(args)




    


