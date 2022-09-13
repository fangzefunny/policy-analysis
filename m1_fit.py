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
parser.add_argument('--data_set',   '-d', help='which_data', type = str, default='exp1data')
parser.add_argument('--method',     '-m', help='methods, mle or map', type = str, default='map')
parser.add_argument('--group',      '-g', help='fit to ind or fit to the whole group', type=str, default='ind')
parser.add_argument('--agent_name', '-n', help='choose agent', default='MixPol')
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
if not os.path.exists(f'{path}/fits/{args.data_set}/{args.agent_name}'):
    os.mkdir(f'{path}/fits/{args.data_set}/{args.agent_name}')

def fit_parallel(pool, data, subj, verbose, args):
    '''A worker in the parallel computing pool 
    '''
    ## fix random seed 
    seed = args.seed
    n_params = args.agent.n_params

    ## Start fitting
    # fit cross validate 
    m_data = np.sum([data[key].shape[0] 
                        for key in data.keys()])
    results = [pool.apply_async(subj.fit, 
                    args=(data, args.method, seed+2*i, verbose)
                    ) for i in range(args.n_fit)]
    opt_nll   = np.inf 
    for p in results:
        params, loss = p.get()
        if loss < opt_nll:
            opt_nll, opt_params = loss, params  
            aic = n_params*2 + 2*opt_nll
            bic = n_params*m_data + 2*opt_nll
    fit_mat = np.hstack([opt_params, opt_nll, aic, bic]).reshape([1, -1])
        
    ## Save the params + nll + aic + bic 
    col = args.agent.p_name + ['nll', 'aic', 'bic']
    print(f'   loss: {fit_mat[0, -3]:.4f}')
    fit_res = pd.DataFrame(fit_mat, columns=col)
    
    return fit_res 

def fit(pool, data, args):
    '''Find the optimal free parameter for each model 
    '''
    ## Define the RL model 
    subj = model(args.agent)

    ## fit list
    fname = f'{save_dir}/fitted_subj_lst-{args.data_set}-{args.agent_name}-{args.method}.csv'
    if os.path.exists(fname):
        fitted_sub = pd.read_csv(f'{fname}') 
        fitted_sub_lst = list(fitted_sub['sub_id'].values)
    else:
        fitted_sub_lst = []
    new_sub_lst = []

    ## Start 
    start_time = datetime.datetime.now()
    
    ## Fit params to each individual 
    if args.group == 'ind':
        done_subj = 0
        all_subj  = len(data.keys()) - len(fitted_sub_lst)
        for sub_idx in data.keys(): 
            if sub_idx not in fitted_sub_lst:  
                print(f'Fitting {args.agent_name} subj {sub_idx}, progress: {(done_subj*100)/all_subj:.2f}%')
                fit_res = fit_parallel(pool, data[sub_idx], subj, False, args)
                pname = f'{save_dir}/{args.agent_name}/params-{args.data_set}-{sub_idx}-{args.method}.csv'
                fit_res.to_csv(pname)
                new_sub_lst.append(sub_idx)
                done_subj += 1
                # save the fitted subjects
                fname = f'{save_dir}/fitted_subj_lst-{args.data_set}-{args.agent_name}-{args.method}.csv'
                df = pd.DataFrame(data={'sub_id': fitted_sub_lst + new_sub_lst})
                df.to_csv(fname)

    ## Fit params to the population level
    elif args.group == 'avg':
        fit_res = fit_parallel(data, pool, subj, True, args)
        pname = f'{save_dir}/params-{args.data_set}-avg.csv'
        fit_res.to_csv(pname)
    
    ## END!!!
    end_time = datetime.datetime.now()
    print('\nparallel computing spend {:.2f} seconds'.format(
            (end_time - start_time).total_seconds()))

def summary(data, args):

    ## Prepare storage
    n_sub    = len(data.keys())
    n_params = args.agent.n_params
    res_mat  = np.zeros([n_sub, n_params+3]) + np.nan 
    res_smry = np.zeros([2, n_params+3]) + np.nan 
    folder   = f'{save_dir}/{args.agent_name}'

    ## Loop to collect data 
    for i, sub_idx in enumerate(data.keys()):
        fname = f'{folder}/params-{args.data_set}-{sub_idx}-{args.method}.csv'
        log = pd.read_csv(fname, index_col=0)
        res_mat[i, :] = log.iloc[0, :].values
        if i == 0: col = log.columns
    
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