import os 
import pickle
import argparse 
import subprocess

import numpy as np 
import pandas as pd

from utils.parallel import get_pool 
from utils.model import *
from utils.env_fn import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',   '-d', help='which_data', type=str, default='exp1data')
parser.add_argument('--env_name',   '-e', help='which enviroment', type=str, default='rl_reversal')
parser.add_argument('--agent_name', '-n', help='choose agent', default='MOS6')
parser.add_argument('--method',     '-m', help='fitting methods', type=str, default='map')
parser.add_argument('--algorithm',  '-a', help='fitting algorithm', type = str, default='BFGS')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=20)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=42)
parser.add_argument('--params',     '-p', help='params', type=str, default='')
parser.add_argument('--recovery',   '-r', help='recovery', type=int, default=1)
args = parser.parse_args()
args.agent = eval(args.agent_name)
env = eval(args.env_name)

# find the current path
pth = os.path.dirname(os.path.abspath(__file__))
dirs = [f'{pth}/simulations', f'{pth}/simulations/{args.data_set}', 
        f'{pth}/simulations/{args.data_set}/{args.agent_name}']
for d in dirs:
    if not os.path.exists(d): os.mkdir(d)

# --------------------------------------------------- #
#     Simulate to understand the three strategies     #
# --------------------------------------------------- #

def sim_three_strategies_paral(pool, n_samples=100, seed=42):
    '''Simulate to understand the three strategies
    '''
    
    # get the representative participants for the HC and PAT groups
    fname = 'fits/exp1data/fit_sub_info-MOS6-map.pkl'
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    fname = 'data/participant_lst.pkl'
    with open(fname, 'rb')as handle: sub_lst = pickle.load(handle)
    rparams = {'general': [], 'HC': [], 'PAT': []}
    for sub_id, item in fit_sub_info.items():
        tmp = [f(p) for p, f in zip(item['param'], MOS6.p_trans)]
        rparams['general'].append(tmp)
        if sub_id in sub_lst['HC']:
            rparams['HC'].append(tmp)
        else:
            rparams['PAT'].append(tmp)

    # get the population median for the first three parameters,
    # and within-group median for the last 3 decision preference parameters
    params = np.median(np.vstack(rparams['general']), axis=0)
    HC_params  = np.hstack([params[:3], np.median(np.vstack(rparams['HC']), axis=0)[3:]])
    PAT_params = np.hstack([params[:3], np.median(np.vstack(rparams['PAT']), axis=0)[3:]])
    HC_params  = np.array([f(p) for f, p in zip(MOS6.p_links, HC_params)])
    PAT_params = np.array([f(p) for f, p in zip(MOS6.p_links, PAT_params)])
    param_lst = [HC_params]*n_samples+[PAT_params]*n_samples
    group_lst = ['HC']*n_samples+['PAT']*n_samples

    # run 100 simulations for each group 
    res = [pool.apply_async(get_sim_sample, 
                            args=(param_lst[i], group_lst[i], 
                                  seed+2*i, i))
                            for i in range(2*n_samples)]
    sim_data_all = pd.concat([p.get() for p in res], axis=0, ignore_index=True)
    fname = f'{pth}/simulations/'
    fname += f'{args.data_set}/MOS6/sim-{args.method}.csv'
    sim_data_all.to_csv(fname)

def get_sim_sample(params, group, seed, sub_id, sim_sample=2):
    model = wrapper(MOS6, env_fn=env)
    sim_data = [] 
    rng = np.random.RandomState(seed)
    i = 0
    for _ in range(sim_sample): 
        for seq_type in ['sta_vol', 'vol_sta']:
            task = env(seq_type).instan(seed+i)
            sim_sample = model.sim_block(task, params, rng=rng)
            sim_sample['block_id'] = f'block_{i}'
            sim_sample['sub_id']   = f'sub_{sub_id}'
            sim_sample['group']    = group
            i += 1
            sim_data.append(sim_sample)
    return pd.concat(sim_data, axis=0, ignore_index=True)

# --------------------------------------------------- #
#      Explain learning rate using the MOS model      #
# --------------------------------------------------- #

def fit_flr_to_mos(n_block=1, n_sub=5, seed=42):
    '''
        Get 2 runs (4 task sequnece) for each participant
    '''
    fname = f'{pth}/data/exp1data.pkl'
    with open(fname, 'rb')as handle: data=pickle.load(handle)
    fname = f'{pth}/fits/exp1data/fit_sub_info-MOS6-map.pkl' 
    with open(fname, 'rb')as handle: fit_info=pickle.load(handle)
    param_mat = [item['param'] for _, item in fit_info.items()]
    params = np.median(param_mat, 0)
    HC_params = params.copy()
    HC_params[3] =  1.5
    HC_params[4] = -1.5
    PAT_params = params.copy()
    PAT_params[3] = -1.5
    PAT_params[4] =  1.5
    mos_data = {}
    mos6_param = {}
    params = { 'HC': HC_params.tolist(), 
              'PAT': PAT_params.tolist()}
    sub_lst = [f'HC{i+1}' for i in range(n_sub)] + [f'PAT{i+1}' for i in range(n_sub)]
    group_lst = ['HC']*n_sub + ['PAT']*n_sub
    seed_idx = 0
    for sub_id, g in zip(sub_lst, group_lst):
        param = params[g]
        syn_data = syn_mos_data(param, sub_id, g, data, seed+seed_idx, n_block=n_block)
        mos_data[sub_id] = syn_data
        mos6_param[sub_id] = param
        seed_idx += 1
    fname = f'{pth}/data/syn_mos6_{n_sub}-sub_param.csv'
    pd.DataFrame.from_dict(mos6_param).to_csv(fname)
    fname = f'{pth}/data/syn_mos6_{n_sub}-sub.pkl'
    with open(fname, 'wb') as handle: pickle.dump(mos_data, handle)

    ## STEP 2: REFIT THE MODEL TO THE SYTHESIZE DATA 
    cmand = ['python', 'm1_fit.py', f'-d=syn_mos6_{n_sub}-sub',
              f'-n=FLR22', '-s=420', '-f=40',
              '-c=40', f'-m="map"', f'-a="BFGS"']
    subprocess.run(cmand)

def syn_mos_data(param, sub_id, group, data, seed, n_block=10):

    # create random state 
    rng = np.random.RandomState(seed)
    model = wrapper(MOS6, env_fn=env)
    ind = rng.choice(list(data.keys()), size=n_block)
    syn_data = {}
    j = 0 
    for idx in ind:
        for i, block_data in data[idx].items():
            sim_data = model.sim({i: block_data}, param, rng)
            sim_data = sim_data.drop(columns=MOS6.voi)
            sim_data['group'] = group
            sim_data['sub_id'] = sub_id
            syn_data[j] = sim_data
            j += 1

    return syn_data

if __name__ == '__main__':

    # STEP1: simulate to understand the three strategies
    pool = get_pool(args)
    sim_three_strategies_paral(pool, args)
    pool.close()

    # STEP 2: explain the FLR using the MOS
    fit_flr_to_mos(args, n_block=2, n_sub=20)

