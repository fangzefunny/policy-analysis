
import os 
import pickle

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import pearsonr

from utils.agent import *
from utils.analyze import *
from utils.model import model 
from utils.viz import viz
viz.get_style()

# set up path 
path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')

feedback_types = ['gain', 'loss']
patient_groups = ['HC', 'PAT']
block_types    = ['sta', 'vol']
policies       = ['EU', 'MO', 'HA']

# fit performance 
def quantTable(agents= ['MixPol', 'GagModel', 'RlRisk']):
    crs = {}
    fname = f'{path}/data/exp1data.pkl'
    with open(fname, 'rb') as handle: data=pickle.load(handle)
    for m in agents:
        subj = model(eval(m))
        n_params = eval(m).n_params
        nll, aic = 0, 0 
        fname = f'{path}/fits/exp1data/fitted_subj_lst-exp1data-GagModel-map.csv'
        subj_Lst = list(pd.read_csv(fname)['sub_id'])
        for sub_id in subj_Lst:
            fname = f'{path}/fits/exp1data/{m}/params-exp1data-{sub_id}-map.csv'
            params = pd.read_csv(fname, index_col=0).iloc[0, 0:n_params].values
            eval_data = subj.sim(data[sub_id], params, rng=None)
            nll += eval_data.loc[:, 'logLike'].sum() / len(subj_Lst)
        aic = 2*nll + 2*n_params
        crs[m] = {'nll':nll, 'aic':aic}

    for m in agents:
        print(f'{m} nll: {crs[m]["nll"]:.3f}, aic: {crs[m]["aic"]:.3f}')

def viz_Human():

    # load data 
    fname = f'{path}/data/exp1_data.csv'
    data  = pd.read_csv(fname)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data['rawRew'] = data.apply(
            lambda x: x[f'mag{int(x["humanAct"])}']
                        *(x["humanAct"]==x["state"]), axis=1)
    data = data.groupby(by=['sub_id', 'b_type', 'feedback_type']).mean().reset_index()

    fig, axs = plt.subplots(2, 2, figsize=(8, 7), sharey=True)
    ax=axs[0, 0]
    for_title = t_test(data, 'is_PAT==True', 'is_PAT==False', tar=['rew'])
    sns.boxplot(x='is_PAT', y='rew', data=data, width=.65,
                    palette=viz.Palette, ax=ax)
    ax.set_xlim([-.8, 1.8])
    ax.set_ylabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HC', 'PAT'])
    ax.set_xlabel('')
    ax.set_title(f'Avg. Reward {for_title[0]}')
    ax=axs[0, 1]
    sns.boxplot(x='b_type', y='rew', data=data,  width=.65,
                    palette=viz.Palette2, ax=ax)
    for_title = t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=['rew'])
    ax.set_xlim([-.8, 1.8])
    ax.set_ylim([0.18, 0.65])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Stable', 'Volatile'])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(f'Avg. Reward {for_title[0]}')
    ax=axs[1, 0]
    sns.boxplot(x='is_PAT', y='rew', data=data,  width=.65,
                    hue='b_type', palette=viz.Palette2, ax=ax)
    #for_title = t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=['rew'])
    ax.set_xlim([-.8, 1.8])
    ax.set_ylim([0.18, 0.65])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['HC', 'PAT'])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.legend(bbox_to_anchor=(1.5, .7), loc='right')
    ax.set_title(f'Avg. Reward ')
    ax=axs[1, 1]
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig1_Human_data.png', dpi=300)

def LR_effect():

    feedback_types = ['gain', 'loss']
    groups  = [0, 1]
    tar    = ['log_alpha']

    nr, nc = len(groups), len(feedback_types)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharex='row')
    for idx, f_type in enumerate(feedback_types):
        for j, is_pat in enumerate(groups):
            ax  = axs[j, idx]
            data = build_pivot_table('map', agent='MixPol', min_q=.01, max_q=.99)
            data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
            data = data.query(f'is_PAT=={is_pat} & feedback_type=="{f_type}"').groupby(by=['sub_id', 'b_type']).mean().reset_index()
            print(f'---------{is_pat}: {f_type}')
            ymin, ymax = data[tar[0]].min(), data[tar[0]].max()
            t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=tar)
            sns.boxplot(x='b_type', y=tar[0], data=data, width=.65,
                            palette=viz.Palette, ax=ax)
            ax.set_xlim([-.8, 1.8])
            ax.set_xticks([0, 1])
            ax.set_ylim([ymin-abs(ymin)*.2, ymax+abs(ymax-1)*.5])
            ax.set_xticklabels(['Stable', 'Volatile'])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_box_aspect(1)
        
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path}/figures/Fig0_LR.png', dpi=300)

                
def HC_PAT_policy():

    tar    = ['l1', 'l2', 'l3']

    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'feedback_type', 'b_type']).mean().reset_index()
    
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharey=True, sharex=True)
    for_title = t_test(data, 'is_PAT==False', 'is_PAT==True', tar=tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x='is_PAT', y=f'{tar[idx]}', data=data, width=.65,
                        palette=viz.Palette, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylim([-5, 5.8])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'PAT'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_box_aspect(1)
        #ax.set_title(f'{titles[idx]}')
        #ax.set_title(f'{titles[idx]} {for_title[idx]}')
        # if idx == 1: ax.legend(bbox_to_anchor=(1.4, 0), loc='lower right')
        # else: ax.get_legend().remove()
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path}/figures/Fig1_HC-PAT-policies.png', dpi=300)

def Policy_Rew():

    tar    = ['l1', 'l2', 'l3']
    titles = [r'$logit(w_{\text{EU}})$: EU', r'$logit(w_2)$: MO', r'$logit(w_3)$: HA']

    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data['rew'] = data['rew'].apply(lambda x: x*100)
    data['rawRew'] = data['rawRew'].apply(lambda x: x*100)
    xmin, xmax = -4.9, 4.9 
    #data[tar].min().min()-.1, data[tar].max().max()+.1

    nr, nc = 2, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharey='row', sharex='col')

    for i, lamb in enumerate(tar):
        for j, feedback_type in enumerate(['gain', 'loss']):
            sel_data = data.query(f'feedback_type=="{feedback_type}"').groupby(
                            by=['sub_id', 'b_type']).mean().reset_index()
            x = sel_data[lamb]
            y = sel_data['rawRew']
            corr, pval = pearsonr(x.values, y.values)
            x = sm.add_constant(x)
            res = sm.OLS(y, x).fit()
            #print(res.summary())
            print(f' {feedback_type}-{lamb}: r={corr:.2f}, p={pval:.3f}')
            #regress = lambda x: res.params['const'] + res.params[lamb]*x

            ax  = axs[j, i]
            x = np.linspace(xmin, xmax, 100)
            sns.scatterplot(x=lamb, y='rawRew', data=sel_data, s=100, 
                                color=viz.Palette2[i], ax=ax)
            ax.set_xlim([-4.5, 4.5]) # for the regression predictor 
            sns.regplot(x=lamb, y='rawRew', data=sel_data, truncate=False,
                            color=[.2, .2, .2], scatter=False, ax=ax)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_xlim([-5, 5])
            ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig2_policies-Rew.png', dpi=300)

def reg(pred='l1', tar='rew'):

    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id']).mean().reset_index()

    x = data[pred]
    y = data[tar]
    x = sm.add_constant(x)
    res = sm.OLS(y, x).fit()
    corr, _ = pearsonr(x, y)
    print(corr)
    reg = lambda x: res.params['const'] + res.params[pred]*x

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    xmin, xmax = x.min().values[1]-.1, x.max().values[1]+.1
    x = np.linspace(xmin, xmax, 100)
    sns.scatterplot(x=pred, y=tar, data=data, 
                        color=viz.Red, ax=ax)
    sns.lineplot(x=x, y=reg(x), color=viz.dBlue, lw=2, ax=ax)
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel(r'$\lambda_1$')
    ax.set_ylabel('rewarding')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig2_{pred}-{tar}.png', dpi=300)

def pred_biFactor():

    preds = ['g', 'g', 'g']
    tars  = ['l1', 'l2', 'l3']
    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id']).mean().reset_index()
   
    nr, nc = 1, int(len(tars))
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharex='row',)
    for i, (pred, tar) in enumerate(zip(preds, tars)):
        xmin, xmax = data[pred].values.min()-.4, data[pred].values.max()+.4
        x = data[pred]
        y = data[tar]
        corr, pval = pearsonr(x.values, y.values)
        x = sm.add_constant(x)
        res = sm.OLS(y, x).fit()
        #print(res.summary())
        print(f' {tar}: r={corr}, p={pval}')
        
        ax  = axs[i]
        x = np.linspace(xmin, xmax, 100)
        sns.scatterplot(x=pred, y=tar, data=data, s=100, 
                            color=viz.Palette2[i], ax=ax)
        ax.set_xlim([xmin, xmax]) # for the regression predictor 
        sns.regplot(x=pred, y=tar, data=data, truncate=False,
                        color=[.2, .2, .2], scatter=False, ax=ax)
        ax.set_ylabel(tar)
        ax.set_xlabel(pred)
        ax.set_xlim([xmin, xmax])
        #ax.set_ylim([-3., 3.])
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig3_pre_syndrome.png', dpi=300)


def pi_effect():

    fname = f'{path}/simulations/MixPol/simsubj-gain_exp1data-sta_first.csv'
    data = pd.read_csv(fname)

    data = data.groupby(by=['trials'])[['l1_effect', 'l2_effect', 'l3_effect']].mean()
    psi  = np.zeros([180])
    psi[:90]     = .7
    psi[90:110]  = .2
    psi[110:130] = .8
    psi[130:150] = .2
    psi[150:170] = .8
    psi[170:180] = .2

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax = ax 
    labels = ['EU', 'MO', 'HA']
    for i in range(3):
        sns.lineplot(x='trials', y=f'l{int(i)+1}_effect', lw=3, 
                    data=data, color=viz.Palette2[i], label=labels[i])
    sns.lineplot(x=np.arange(180), y=psi, color='k', ls='--')
    ax.set_ylabel('Prob. of choosing \nthe left stimulus')
    ax.set_xlabel('Trials')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{path}/figures/effect.png', dpi=300)


if __name__ == '__main__':

    #quantTable(['MixPol', 'GagModel'])
    #viz_Human()
    LR_effect()
    #viz_PiReward()
    HC_PAT_policy()
    Policy_Rew()
    pred_biFactor()
    #pi_effect()