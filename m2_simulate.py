import os 
import pickle
import argparse 

import pandas as pd

from utils.parallel import get_pool 
from utils.model import model
from utils.analyze import build_pivot_table
from utils.agent import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

## pass the hyperparams
parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp1data')
parser.add_argument('--method',     '-m', help='fitting methods', type = str, default='bms')
parser.add_argument('--group',      '-g', help='fit to ind or fit to the whole group', type=str, default='ind')
parser.add_argument('--agent_name', '-n', help='choose agent', default='MOS_fix')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--n_sim',      '-f', help='f simulations', type=int, default=5)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=120)
parser.add_argument('--params',     '-p', help='params', type=str, default='')
parser.add_argument('--recovery',   '-r', help='recovery', type=int, default=1)
args = parser.parse_args()
args.agent = eval(args.agent_name)

# create the folders for this folder
if not os.path.exists(f'{path}/simulations'):
    os.mkdir(f'{path}/simulations')
if not os.path.exists(f'{path}/simulations/{args.data_set}'):
    os.mkdir(f'{path}/simulations/{args.data_set}')
if not os.path.exists(f'{path}/simulations/{args.data_set}/{args.agent_name}'):
    os.mkdir(f'{path}/simulations/{args.data_set}/{args.agent_name}')

# --------- Simulate for Analysis ---------- #

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
    fname = f'{path}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'      
    with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
    for sub_idx in data.keys(): 
        if in_params is None:
            n_params = args.agent.n_params
            if args.group == 'ind': 
                params = fit_sub_info[sub_idx]['param']
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
    subj   = model(MOS_fix)

    # load parameters 
    pTable = build_pivot_table('bms', agent='MOS_fix', verbose=False)
    pTable['group'] = pTable['group'].map({'HC': 'HC', 'GAD': 'PAT', 'MDD': 'PAT'})
    g_param = pTable.groupby(by='group').mean(numeric_only=True)[['alpha', 'l1', 'l2', 'l3']]
    n_params = subj.agent.n_params
    fname    = f'{path}/fits/{args.data_set}/params-{args.data_set}-{args.agent_name}-bms-ind.csv'      
    params   = pd.read_csv(fname, index_col=0).iloc[0, 0:n_params].values
    if mode != 'AVG': params = np.hstack([params[:2], g_param.loc[mode, :].values])

    # start simulation
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
        task['group'] = mode
        
        for j in range(n_samples):
            sim_rng = np.random.RandomState(seed+j)
            sim_sample = subj.sim_block(task, params, rng=sim_rng, is_eval=False)
            sim_data[cond].append(sim_sample)
        
    return sim_data 

# --------- Simulate for recovery ---------- #

def for_param_recovery(data, n_sub=20, n_block=10):

    # set seed
    rng = np.random.RandomState(args.seed+1)

    # init model 
    subj = model(args.agent)

    ## get parameters 
    fname = f'{path}/fits/exp1data/params-exp1data-MOS_fix-bms-ind.csv'
    p1_mu  = pd.read_csv(fname, index_col=0).iloc[0, 0:2].values
    p1_sig = pd.read_csv(fname, index_col=0).iloc[1, 0:2].values
    
    ## get the mean parameters from the fit results 
    m_types = ['vary_w', 'vary_lr']
    groups  = ['HC', 'PAT']
    pTable = build_pivot_table('bms', agent='MOS_fix')
    pTable['group'] = pTable['group'].map({'HC': 'HC', 'GAD': 'PAT', 'MDD': 'PAT'})

    cases = {'vary_w' :{'HC':{}, 'PAT':{}}, 
            'vary_lr':{'HC':{}, 'PAT':{}}}

    ## get param for vary weights
    for g_type in groups:
        lst = pTable.groupby(by=['group']
                        ).mean(numeric_only=True).reset_index(
                        ).query(f'group=="{g_type}"').loc[:, 
                        ['alpha', 'l1', 'l2', 'l3']].values.reshape([-1])
        cases['vary_w'][g_type]['mu'] = np.hstack([p1_mu, lst])
        
        lst = pTable.groupby(by=['group']
                        ).std(numeric_only=True).reset_index(
                        ).query(f'group=="{g_type}"').loc[:, 
                        ['alpha', 'l1', 'l2', 'l3']].values.reshape([-1])
        cases['vary_w'][g_type]['sig'] = np.hstack([p1_sig, lst])

    ## get param for vary weights
    for g_type, lr in zip(groups, [.2, .05]):
        lst = pTable.loc[:, ['l1', 'l2', 'l3']].mean().values.reshape([-1])
        cases['vary_lr'][g_type]['mu'] = np.hstack([p1_mu, lr, lst])
        
        lst = pTable.loc[:, ['l1', 'l2', 'l3']].std().values.reshape([-1])
        cases['vary_lr'][g_type]['sig'] = np.hstack([p1_sig, .05, lst])

    truth_params = {p: [] for p in subj.agent.p_name}
    truth_params['m_type'] = []
    truth_params['group']  = [] 
    truth_params['sub_id'] = []

    sim_id = 0
    for m_type in m_types:
        for g_type in groups:
            HC_var  = [norm(loc=cases[m_type][g_type]['mu'][i],  
                        scale=cases[m_type][g_type]['sig'][i]) 
                        for i in range(subj.agent.n_params)]
            for _ in range(n_sub):
                # load parameters
                for i in range(subj.agent.n_params):
                    param = np.clip(HC_var[i].rvs().round(4), 
                                    subj.agent.bnds[i][0],
                                    subj.agent.bnds[i][1])
                    truth_params[subj.agent.p_name[i]].append(param)
                truth_params['m_type'].append(m_type)
                truth_params['group'].append(g_type)
                truth_params['sub_id'].append(sim_id)
                sim_id += 1

    ## save the ground turth parameters             
    truth_params_lst = pd.DataFrame.from_dict(truth_params)
    fname = f'{path}/data/params_truth-{args.data_set}-{args.agent_name}.csv'
    truth_params_lst.to_csv(fname)
    
    ## start simulate with the generated parameters  
    data_for_recovery = {}
    for i, row in truth_params_lst.iterrows():
        # get task 
        ind = rng.choice(list(data.keys()), size=n_block)
        param = list(row[subj.agent.p_name].values)
        data_for_recovery[row['sub_id']] = {}
        for idx in ind:
            sim_data = subj.sim({0: data[idx][list(data[idx].keys())[0]]}, param, rng)
            sim_data['humanAct'] = sim_data['act'].astype(int)
            sim_data = sim_data.drop(columns=['ps', 'pi', 'alpha', 'acc',
            'w1', 'w2', 'w3', 'l1', 'l2', 'l3', 
            'l1_effect', 'l2_effect', 'l3_effect', 
            'act'])
            data_for_recovery[row['sub_id']][idx] = sim_data
    
    ## save for recovery
    fname = f'{path}/data/param_recovery-{args.agent_name}.pkl'
    with open(fname, 'wb')as handle:
        pickle.dump(data_for_recovery, handle)
    
      
def for_model_recovery_paral(pool, data, n_sub=40):

    # set seed 
    seed = args.seed+2
    rng = np.random.RandomState(seed)

    ## get parameters 
    fname = f'{path}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'      
    with open(fname, 'rb')as handle: fit_sub_info_orig = pickle.load(handle)

    ## create a sub list of subject list 
    new_keys = rng.choice(list(fit_sub_info_orig.keys()), size=n_sub)
    fit_sub_info = {k: fit_sub_info_orig[k] for k in new_keys}

    res = [pool.apply_async(for_model_recovery, args=(sub_idx, data, fit_sub_info, seed+5*i))
                            for i, sub_idx in enumerate(fit_sub_info.keys())]
    syn_data = {}
    for _, p in enumerate(res):
        sub_idx, sim_data = p.get() 
        syn_data[sub_idx] = sim_data 

    # save for fit 
    with open(f'{path}/data/{args.data_set}-{args.agent_name}.pkl', 'wb')as handle:
        pickle.dump(syn_data, handle)
    print(f'Synthesize data for {args.agent_name} has been saved!')


def for_model_recovery(sub_idx, data, fit_sub_info, seed, n_sample=10):

    # init model 
    subj = model(args.agent)
    rng = np.random.RandomState(seed)

    # synthesize the data and save
    sim_data = {} 
    task_ind = rng.choice(list(data.keys()), size=n_sample)
    param = fit_sub_info[sub_idx]['param']
    for i, task_idx in enumerate(task_ind):
        task = data[task_idx][list(data[task_idx].keys())[0]]
        sim_sample = subj.sim({i: task}, param, rng=rng)
        sim_sample['humanAct'] = sim_sample['act'].astype(int)
        sim_sample = sim_sample.drop(columns=['ps', 'pi', 'alpha', 'acc',
                'w1', 'w2', 'w3', 'l1', 'l2', 'l3', 
                'l1_effect', 'l2_effect', 'l3_effect', 
                'act'])
        sim_data[i] = sim_sample
     
    return sub_idx, sim_data

if __name__ == '__main__':
    
    ## STEP 0: GET PARALLEL POOL
    print(f'Simulating {args.agent_name}')
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{path}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)

    ## STEP 2: SYNTHESIZE DATA
    sim_paral(pool, data, args)

    # STEP 3: SIM FOR RECOVERY
    if args.recovery: 
        for_model_recovery_paral(pool, data)
        pool.close()
        if (args.agent_name=='MOS_fix') and (args.data_set=='exp1data'): 
            for_param_recovery(data)

    # # STEP 4: SIM SUBJECT
    # if (args.agent_name=='MOS_fix') and (args.data_set=='exp1data'): 
    #     for m in ['HC', 'PAT', 'AVG']: sim_subj_paral(pool, m, args)

        

    

   
