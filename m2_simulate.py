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
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp1data')
parser.add_argument('--method',     '-m', help='fitting methods', type = str, default='map')
parser.add_argument('--group',      '-g', help='fit to ind or fit to the whole group', type=str, default='ind')
parser.add_argument('--agent_name', '-n', help='choose agent', default='MOS')
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
if not os.path.exists(f'{path}/simulations/{args.data_set}'):
    os.mkdir(f'{path}/simulations/{args.data_set}')
if not os.path.exists(f'{path}/simulations/{args.data_set}/{args.agent_name}'):
    os.mkdir(f'{path}/simulations/{args.data_set}/{args.agent_name}')

# define functions
def simulate(data, args, seed):

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
                fname = f'{path}/fits/{args.data_set}/{args.agent_name}/params-{args.data_set}-{sub_idx}-{args.method}.csv'      
            elif args.group == 'avg':
                fname = f'{path}/fits/{args.data_set}/params-{args.data_set}-{args.method}-{args.agent_name}-avg.csv'      
            params = pd.read_csv(fname, index_col=0).iloc[0, 0:n_params].values
        else:
            params = in_params
        
        # synthesize the data and save
        rng = np.random.RandomState(seed)
        sim_sample = subj.sim(data[sub_idx], params, rng=rng)
        sim_data.append(sim_sample)
        seed += 1

    return pd.concat(sim_data, axis=0)

# define functions
def sim_paral(pool, data, args):
    
    ## Simulate data for n_sim times 
    seed = args.seed 
    res = [pool.apply_async(simulate, args=(data, args, seed+5*i))
                            for i in range(args.n_sim)]
    for i, p in enumerate(res):
        sim_data = p.get() 
        fname = f'{path}/simulations/{args.data_set}/{args.agent_name}/'
        fname += f'sim-{args.data_set}-{args.method}-idx{i}.csv'
        sim_data.to_csv(fname, index = False, header=True)

def sim_subj_paral(pool, mode, args, n_sim=500):

    res = [pool.apply_async(sim_subj, args=[mode, args.seed+i])
                            for i in range(n_sim)]
    sim_sta_first = []
    sim_vol_first = [] 
    for _, p in enumerate(res):
        sim_data = p.get() 
        sim_sta_first += sim_data['sta_first']
        sim_vol_first += sim_data['vol_first']
    
    fname = f'{path}/simulations/{args.data_set}/{args.agent_name}/simsubj-{args.data_set}-sta_first-{mode}.csv'
    sim_sta_first = pd.concat(sim_sta_first, ignore_index=True)
    sim_sta_first.to_csv(fname, index = False, header=True)
    fname = f'{path}/simulations/{args.data_set}/{args.agent_name}/simsubj-{args.data_set}-vol_first-{mode}.csv'
    sim_vol_first = pd.concat(sim_vol_first, ignore_index=True)
    sim_vol_first.to_csv(fname, index = False, header=True)

def get_data(rng, n_trials=180, sta_first=True):
    psi    = np.zeros(n_trials)
    state  = np.zeros(n_trials)
    if sta_first:
        psi[:90]     = .7
        psi[90:110]  = .2
        psi[110:130] = .8
        psi[130:150] = .2
        psi[150:170] = .8
        psi[170:180] = .2
        b_type       = ['sta']*90 + ['vol']*90
    else:
        psi[:20]     = .2
        psi[20:40]   = .8
        psi[40:60]   = .2
        psi[60:80]   = .8
        psi[80:90]   = .2
        psi[90:]     = .7
        b_type       = ['vol']*90 + ['sta']*90
    
    for i in range(n_trials):
        if rng.rand(1) < psi[i]:
            state[i] = 1

    return state, psi, b_type

def sim_subj(mode, seed, n_samples=3):
       
    # decide what to collect
    subj   = model(MixPol)
    if mode == 'HC':
        ls = [0.678336, -0.976054, 0.297795]
    elif mode == 'PAT':
        ls = [0.007705, 0.315989, -0.323734]
    n_params = 18
    fname    = f'{path}/fits/{args.data_set}/params-{args.data_set}-{args.agent_name}-map-ind.csv'      
    params   = pd.read_csv(fname, index_col=0).iloc[0, 0:n_params].values
    params[3:6]   = ls
    params[7:10]  = ls
    params[11:14] = ls
    params[15:18] = ls 
        
    rng    = np.random.RandomState(seed)
    
    # simulate block n times
    sim_data = {'sta_first': [], 'vol_first': []}
    for i, cond in enumerate(['sta_first', 'vol_first']):
        m1 = np.linspace(0, 1, 180).round(2)
        rng.shuffle(m1)
        m2 = np.linspace(0, 1, 180).round(2)
        rng.shuffle(m2)
        state, psi, b_type = get_data(rng, sta_first=(1-i))
        task = {
            'mag0':   m1,
            'mag1':   m2,
            'b_type': b_type,
            'state':  state.astype(int),
            'psi':    psi,
            'trials': list(range(180)),
            'feedback_type': ['gain']*180,
        }
        task = pd.DataFrame.from_dict(task)
        
        for j in range(n_samples):
            sim_rng = np.random.RandomState(seed+j)
            sim_sample = subj.sim_block(task, params, rng=sim_rng, is_eval=False)
            sim_data[cond].append(sim_sample)
        
    return sim_data 
            

if __name__ == '__main__':
    
    ## STEP 0: GET PARALLEL POOL
    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{path}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

    # STEP 2: SYNTHESIZE DATA
    #sim_paral(pool, data, args)

    # STEP 3: SIM SUBJECT
    sim_subj_paral(pool, 'HC', args)
    sim_subj_paral(pool, 'PAT', args)
