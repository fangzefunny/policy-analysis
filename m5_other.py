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

def sim_paral(pool, args, n_samples=100):
    alphas = np.linspace(.1, .8, 10)
    betas  = np.linspace(.5,  6, 10)
    rng = np.random.RandomState(1234)
    param_lst = []
    for _ in range(n_samples):
        alpha = rng.choice(alphas)
        beta  = rng.choice(betas)
        params = np.array([beta, alpha])
        param_lst.append(params.copy())
    res = [pool.apply_async(get_sim_sample, 
                            args=(params, args.seed+2*i, i))
                            for i, params in enumerate(param_lst)]
    sim_data_all = pd.concat([p.get() for p in res], axis=0, ignore_index=True)
    fname = f'{pth}/simulations/'
    fname += f'{args.data_set}/EU/sim-{args.method}.csv'
    sim_data_all.to_csv(fname)

def get_sim_sample(params, seed, sub_id, sim_sample=3):
    model = wrapper(EU, env_fn=env)
    sim_data = [] 
    rng = np.random.RandomState(seed)
    for i in range(sim_sample): 
        task = env().instan(seed+i)
        sim_sample = model.sim_block(task, params, rng=rng)
        sim_sample['beta']     = params[0]
        sim_sample['alpha']    = params[1]
        sim_sample['block_id'] = f'block_{i}'
        sim_sample['sub_id']   = f'sub_{sub_id}'
        sim_data.append(sim_sample)
    return pd.concat(sim_data, axis=0, ignore_index=True)

def eval_model(args):
    model = wrapper(args.agent, env_fn=env)
    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb')as handle: data=pickle.load(handle)
    # load parameter
    fname = f'{pth}/fits/{args.data_set}/fit_sub_info'
    fname += f'-{args.agent_name}-{args.method}.pkl'
    with open(fname, 'rb')as handle: fit_info=pickle.load(handle)
    sub_lst = list(data.keys())
    # evaluate the model
    sim_data = []
    for sub_id in sub_lst:
        params = fit_info[sub_id]['param']
        sim_datum = model.eval(data[sub_id], params)
        sim_data.append(sim_datum)
    return pd.concat(sim_data, axis=0, ignore_index=True)

def fit_flr_to_mos(args, n_block=5, n_sub=10):

    fname = f'{pth}/data/{args.data_set}.pkl'
    with open(fname, 'rb')as handle: data=pickle.load(handle)
    fname = f'{pth}/simulations/exp1data/MOS6/sim-map.csv' 
    sim_data = pd.read_csv(fname, index_col=0)
    p_name = ['alpha_act', 'beta', 'alpha', 'l1', 'l2', 'l3']
    sim_data['group'] = sim_data['group'].map(
        {'HC': 'HC', 'MDD': 'PAT', 'GAD': 'PAT'}
        )
    sim_data['alpha_act'] = sim_data['alpha_act'].apply(
        lambda x: np.log(x+1e-12)-np.log(1-x+1e-12)
    )
    sim_data['alpha'] = sim_data['alpha'].apply(
        lambda x: np.log(x+1e-12)-np.log(1-x+1e-12)
    )
    sim_data['beta'] = sim_data['beta'].apply(
        lambda x: np.log(x+1e-12)
    )
    group_data = sim_data.groupby(by=['group'])[p_name].mean()
    m_data = group_data.mean(0)
    for p in ['alpha_act', 'beta', 'alpha', 'l3']: group_data[p] = m_data[p]
    group = ['HC', 'PAT']
    mos_data = {}
    mos6_param = {}
    params = { 'HC': [-.4, 2, .8, 1.5, -.5, -1.15], 
              'PAT': [-.4, 2, .8, -.5, 1.5, -1.15]}
    sub_lst = [f'HC{i+1}' for i in range(n_sub)] + [f'PAT{i+1}' for i in range(n_sub)]
    group_lst = ['HC']*n_sub + ['PAT']*n_sub
    seed_idx = 0
    for sub_id, g in zip(sub_lst, group_lst):
        param = params[g]
        syn_data = syn_mos_data(param, sub_id, g, data, args.seed+seed_idx, n_block=n_block)
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
              '-c=40', f'-m={args.method}', f'-a={args.algorithm}']
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

    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)
    sim_paral(pool, args)
    # fit_flr_to_mos(args, n_block=2, n_sub=21)
    # eval_model(args)
    pool.close()