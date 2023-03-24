import os 
import pickle
import argparse 
import datetime 
import numpy as np
import pandas as pd

from utils.parallel import get_pool 
from utils.model import model
from utils.agent import *

# find the current path
path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='Test for argparse')
parser.add_argument('--n_fit',      '-f', help='fit times', type = int, default=1)
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='for_interpret_HC')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='bms')
parser.add_argument('--group',      '-g', help='fit to ind or fit to the whole group', type=str, default='group')
parser.add_argument('--agent_name', '-n', help='choose agent', default='RS_test')
parser.add_argument('--n_cores',    '-c', help='number of CPU cores used for parallel computing', 
                                            type=int, default=1)
parser.add_argument('--seed',       '-s', help='random seed', type=int, default=420)
args = parser.parse_args()
args.agent = eval(args.agent_name)

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
save_dir = f'{path}/fits/{args.data_set}' 


# create the folders if not existed
if not os.path.exists(f'{path}/fits'):
    os.mkdir(f'{path}/fits')
if not os.path.exists(f'{path}/fits/{args.data_set}'):
    os.mkdir(f'{path}/fits/{args.data_set}')


def fit(pool, data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## Define the RL model 
    subj = model(args.agent)

    ## fit list
    fname = f'{path}/fits/{args.data_set}/fit_sub_info-{args.agent_name}-{args.method}.pkl'
    if os.path.exists(fname):
        # load the previous fit resutls
        with open(fname, 'rb')as handle: fit_sub_info = pickle.load(handle)
        fitted_sub_lst = [k for k in fit_sub_info.keys()]
    else:
        fitted_sub_lst = []
        fit_sub_info = {}

    ## Start 
    start_time = datetime.datetime.now()
    
    ## Fit params to each individual 
    if args.group == 'ind':
        done_subj = 0
        all_subj  = len(data.keys()) - len(fitted_sub_lst)
        for sub_idx in data.keys(): 
            if sub_idx not in fitted_sub_lst:  
                print(f'Fitting {args.agent_name} subj {sub_idx}, progress: {(done_subj*100)/all_subj:.2f}%')
                fit_info = fit_parallel(pool, data[sub_idx], subj, False, args)
                fit_sub_info[sub_idx] = fit_info
                with open(fname, 'wb')as handle: 
                    pickle.dump(fit_sub_info, handle)
                done_subj += 1
                
    ## Fit params to the population level
    elif args.group == 'group':
        fit_sub_info = fit_parallel(pool, data, subj, False, args)
        with open(fname, 'wb')as handle: 
            pickle.dump(fit_sub_info, handle)
    
    ## END!!!
    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))
        
def fit_parallel(pool, data, subj, verbose, args):
    '''A worker in the parallel computing pool 
    '''
    ## fix random seed 
    seed = args.seed
    n_params = args.agent.n_params

    ## Start fitting
    # fit cross validate 
    if args.group:
        m_data = 0
        for k1 in data.keys():
            for k2 in data[k1].keys():
                m_data += data[k1][k2].shape[0]
    else:
        m_data = np.sum([data[key].shape[0] 
                        for key in data.keys()])
    results = [pool.apply_async(subj.fit, 
                    args=(data, 
                          args.method, 
                          seed+2*i, 
                          False,    
                          verbose,
                          args.group)
                    ) for i in range(args.n_fit)]
    opt_val   = np.inf 
    for p in results:
        res = p.get()
        if res.fun < opt_val:
            opt_val = res.fun
            opt_res = res
            
    ## Save the optimize results 
    fit_res = {}
    fit_res['log_post']   = -opt_val
    fit_res['log_like']   = 0 if args.group == 'group' else \
                            subj.loglike(opt_res.x, data)
    fit_res['param']      = opt_res.x
    fit_res['param_name'] = args.agent.group_p_name \
                            + args.agent.p_name*len(data.keys()
                            ) if args.group == 'group' else args.agent.p_name
    fit_res['n_param']    = n_params
    fit_res['aic']        = n_params*2 - 2*fit_res['log_like']
    fit_res['bic']        = n_params*np.log(m_data) - 2*fit_res['log_like']
    if args.method == 'bms':
        fit_res['H'] = np.linalg.inv(opt_res.hess_inv.todense())

    return fit_res 

def summary(data, args):

    ## Prepare storage
    n_sub    = len(data.keys())
    n_params = args.agent.n_params
    field    = ['log_post', 'log_like', 'aic', 'bic']
    res_mat  = np.zeros([n_sub, n_params+len(field)]) + np.nan 
    res_smry = np.zeros([2, n_params+len(field)]) + np.nan 
    fname =  f'{path}/fits/{args.data_set}/'
    fname += f'fit_sub_info-{args.agent_name}-{args.method}.pkl'

    ## Loop to collect dat 
    with open(fname, 'rb')as handle: 
        fit_sub_info = pickle.load(handle)

    for i, sub_idx in enumerate(fit_sub_info.keys()):
        sub_info = fit_sub_info[sub_idx]
        res_mat[i, :] = np.hstack([sub_info[p] 
                        for p in ['param']+field])
        if i==0: col = sub_info['param_name'] + field
    
    ## Compute and save the mean and sem
    res_smry[0, :] = np.mean(res_mat, axis=0)
    res_smry[1, :] = np.std(res_mat, axis=0) / np.sqrt(n_sub)
    fname = f'{save_dir}/params-{args.data_set}-{args.agent_name}-{args.method}-ind.csv'
    pd.DataFrame(res_smry, columns=col).round(4).to_csv(fname)

if __name__ == '__main__':

    ## STEP 0: GET PARALLEL POOL
    pool = get_pool(args)

    ## STEP 1: LOAD DATA 
    fname = f'{path}/data/{args.data_set}.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
  
    ## STEP 2: FIT
    fit(pool, data, args)
    # summary the mean and std for parameters 
    if args.group=='ind': summary(data, args)