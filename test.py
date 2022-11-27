import os 
import pickle 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from utils.bms import fit_bms
from utils.viz import viz
viz.get_style()

# set up path 
path = os.path.dirname(os.path.abspath(__file__))


# models = ['MOS', 'FLR']
# fit_sub_info = []

# for i, m in enumerate(models):
#     with open(f'fits/exp1data/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
#         fit_info = pickle.load(handle)
#     # get the subject list 
#     if i==0: subj_lst = fit_info.keys() 
#     # get log post
#     log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
#     bic      = [fit_info[idx]['bic'] for idx in subj_lst]
#     h        = [fit_info[idx]['H'] for idx in subj_lst]
#     n_param  = fit_info[list(subj_lst)[0]]['n_param']
#     fit_sub_info.append({
#         'log_post': log_post, 
#         'bic': bic, 
#         'n_param': n_param, 
#         'H': h
#     })

# model recovery

def show_bms(data_set, models = ['MOS', 'FLR', 'RP'], 
             n_param =[18, 15, 9, 6, 6, 3]):
    '''group-level bayesian model selection
    '''
    
    ticks = [f'{m}({n})' for n, m in zip(n_param, models)]
    fit_sub_info = []

    for i, m in enumerate(models):
        with open(f'fits/{data_set}/fit_sub_info-{m}-bms.pkl', 'rb')as handle:
            fit_info = pickle.load(handle)
        # get the subject list 
        if i==0: subj_lst = fit_info.keys() 
        # get log post
        log_post = [fit_info[idx]['log_post'] for idx in subj_lst]
        bic      = [fit_info[idx]['bic'] for idx in subj_lst]
        h        = [fit_info[idx]['H'] for idx in subj_lst]
        n_param  = fit_info[list(subj_lst)[0]]['n_param']
        fit_sub_info.append({
            'log_post': log_post, 
            'bic': bic, 
            'n_param': n_param, 
            'H': h
        })

    bms_results = fit_bms(fit_sub_info)

    # show protected exceedence 
    _, ax = plt.subplots(1, 1, figsize=(5, 4))
    xx = list(range(len(models)))
    sns.barplot(x=xx, y=bms_results['pxp'], palette=viz.Palette[:len(models)], ax=ax)
    ax.set_xticks(xx)
    ax.set_xticklabels(ticks, rotation=45)
    ax.set_xlim([0-.8, len(models)-1+.8])
    ax.set_ylabel('PXP')
    plt.tight_layout()
    plt.savefig(f'{path}/figures/BMS-{data_set}.png', dpi=300)

# show_bms(data_set='exp1data-MOS')
# show_bms(data_set='exp1data-FLR')
# show_bms(data_set='exp1data-RP')

# fit performance 
def quantTable(data_set, agents= ['MOS', 'FLR', 'RP'], n_param =[18, 15, 9,]):
    crs = {}
    ticks = [f'{m}({n})' for n, m in zip(n_param, agents)]

    for i, m in enumerate(agents):
        n_params = n_param[i]
        nll, aic = 0, 0 
        fname = f'{path}/fits/{data_set}/params-{data_set}-{m}-bms-ind.csv'
        data  = pd.read_csv(fname)
        nll   = -data.loc[0, 'log_like']
        aic   = data.loc[0, 'aic']
        bic   = data.loc[0, 'bic']
        crs[m] = {'NLL': nll, 'AIC': aic, 'BIC': bic}

    for m in agents:
        print(f'{m}({n_params}) nll: {crs[m]["NLL"]:.3f}, aic: {crs[m]["AIC"]:.3f}, bic: {crs[m]["BIC"]:.3f}')

    fig, axs = plt.subplots(3, 1, figsize=(6, 11))
    xx = list(range(len(agents)))
    for i, c in enumerate(['NLL', 'AIC', 'BIC']):
        cr = np.array([crs[m][c] for m in agents])
        cr -= cr.min()
        ax = axs[i]
        sns.barplot(x=xx, y=cr, palette=viz.Palette[:len(agents)], ax=ax)
        ax.set_xticks(xx)
        ax.set_xticklabels(ticks, rotation=45)
        ax.set_xlim([0-.8, len(agents)-1+.8])
        ax.set_ylabel(f'Delta {c}')
        plt.tight_layout()
        plt.savefig(f'{path}/figures/quant-{data_set}.png', dpi=300)

quantTable('exp1data-MOS')
quantTable('exp1data-FLR')
quantTable('exp1data-RP')
