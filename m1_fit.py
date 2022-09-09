import os 
import pickle
import argparse 

import pandas as pd

from utils.parallel import get_pool 
from utils.model import model
from utils.agent import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='gain_data')
parser.add_argument('--method',     '-m', help='fitting methods', type = str, default='map')
parser.add_argument('--group',      '-g', help='fit to ind or fit to the whole group', type=str, default='ind')
parser.add_argument('--agent_name', '-n', help='choose agent', default='mix_pol_3w')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--n_sim',      '-f', help='f simulations', type=int, default=5)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=120)
parser.add_argument('--params',     '-p', help='params', type=str, default='')
args = parser.parse_args()
args.agent = eval(args.agent_name)

# create the folders for this folder
if not os.path.exists(f'{path}/simulations'):
    os.mkdir(f'{path}/simulations')
if not os.path.exists(f'{path}/simulations/{args.agent_name}'):
    os.mkdir(f'{path}/simulations/{args.agent_name}')

# define functions
def simulate(data, args, pi_id, seed):

    # define the subj
    subj = model(args.agent)
    n_params = args.agent.n_params 
    # if there is input params 
    if args.params != '': 
        in_params = [float(i) for i in args.params.split(',')]
    else: in_params = None 

    ## Loop to choose the best model for simulation
    # the last column is the loss, so we ignore that
    sim_data = []
    for sub_idx in data.keys(): 
        if in_params is None:
            n_params = args.agent.n_params
            if args.group == 'ind': 
                fname = f'{path}/fits/{args.agent_name}/params-{args.data_set}-{args.method}-{sub_idx}.csv'      
            elif args.group == 'avg':
                fname = f'{path}/fits/params-{args.data_set}-{args.method}-{args.agent_name}-avg.csv'      
            params = pd.read_csv(fname, index_col=0).iloc[0, 0:n_params].values
        else:
            params = in_params

        # assign policy method 
        if args.agent_name == 'MixPol':
            mask  = np.zeros_like(params)
            mask[list(range(4))+[4+pi_id, 7+pi_id]] = 1
            mask = -(1-mask)*1e12 + mask*0
            params += mask 
        
        # synthesize the data and save
        rng = np.random.RandomState(seed)
        sim_sample = subj.sim(data[sub_idx], params, rng=rng)
        sim_data.append(sim_sample)
        seed += 1

    return pd.concat(sim_data, axis=0)

# define functions
def sim_paral(pool, data, args):

    policies = ['EU', 'MO', 'HA']
    nPi = len(policies)
    
    ## Simulate data for n_sim times 
    seed = args.seed 
    res = [pool.apply_async(simulate, args=(data, args, i%nPi, seed+5*i))
                            for i in range(args.n_sim*nPi)]
    for i, p in enumerate(res):
        sim_data = p.get() 
        sim_id, pi_id = i//nPi, i%nPi 
        fname = f'{path}/simulations/{args.agent_name}/sim-{args.data_set}'
        fname += f'-{args.method}-idx{sim_id}-{policies[pi_id]}.csv'
        sim_data.to_csv(fname, index = False, header=True)

def summary(data, args):

    ## Prepare storage
    n_sub    = len(data.keys())
    n_params = args.agent.n_params
    res_mat  = np.zeros([n_sub, n_params+3]) + np.nan 
    res_smry = np.zeros([2, n_params+3]) + np.nan 
    folder   = f'{path}/fits/{args.agent_name}'

    ## Loop to collect data 
    for i, sub_idx in enumerate(data.keys()):
        fname = f'{folder}/params-{args.data_set}-{args.method}-{sub_idx}.csv'
        log = pd.read_csv(fname, index_col=0)
        res_mat[i, :] = log.iloc[0, :].values
        if i == 0: col = log.columns
    
    ## Compute and save the mean and sem
    res_smry[0, :] = np.mean(res_mat, axis=0)
    res_smry[1, :] = np.std(res_mat, axis=0) / np.sqrt(n_sub)
    fname = f'{path}/fits/params-{args.data_set}-{args.agent_name}-{args.method}-ind.csv'
    pd.DataFrame(res_smry, columns=col).round(4).to_csv(fname)
    
if __name__ == '__main__':
    
    ## STEP 0: GET PARALLEL POOL
    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{path}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

    ## STEP 2: SYNTHESIZE DATA
    sim_paral(pool, data, args)
    if args.group=='ind': summary(data, args)
