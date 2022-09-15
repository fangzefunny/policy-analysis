
import os 

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
    
    for m in agents:
        subj = model(eval(m))
        n_params = eval(m).n_params
        nll, aic = 0, 0 
        fname = f'{path}/simulations/exp1data/{m}/sim-exp1data-map-idx0.csv'
        data  = pd.read_csv(fname)
        nll   = data.groupby(by=['sub_id']).sum()['logLike'].mean()
        aic   = 2*nll + 2*n_params
        crs[m] = {'nll':nll, 'aic':aic}

    for m in agents:
        print(f'{m}({eval(m).n_params}) nll: {crs[m]["nll"]:.3f}, aic: {crs[m]["aic"]:.3f}')

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

def LR_effect1():

    tar = ['log_alpha']

    data = build_pivot_table('map', agent='MixPol', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')

    nr, nc = 1, len(feedback_types)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharex='row')
    for i, f_type in enumerate(feedback_types):

        ax = axs[i]
        sel_data = data.query(f'feedback_type=="{f_type}"')
        ymin, ymax = sel_data[tar[0]].min(), sel_data[tar[0]].max()
        t_test(sel_data, 'b_type=="sta"', 'b_type=="vol"', tar=tar)
        sns.boxplot(x='b_type', y=tar[0], data=sel_data, 
                        width=.65, palette=viz.BluePairs, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_xticks([0, 1])
        ax.set_ylim([ymin-abs(ymin)*.2, ymax+abs(ymax-1)*.5])
        ax.set_xticklabels(['Stable', 'Volatile'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig0_LR_effect1.png', dpi=300)

def LR_effect2():
    data = build_pivot_table('map', agent='MixPol', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    vendors  = data['b_type'].unique()
    data = pd.concat([data.set_index(['sub_id', 'is_PAT', 'feedback_type', 'g', 'f1', 'f2']).groupby('b_type'
                    )['log_alpha'].get_group(key) for key in vendors],axis=1)
    data.columns = ['sta', 'vol']
    data['log_diff'] = data['sta'] - data['vol']
    data.reset_index(inplace=True)
    data = data.groupby(by=['sub_id']).mean().reset_index()
    data = data.dropna()

    syndromes = ['g', 'f1', 'f2']
    nr, nc = 1, len(syndromes)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharey=True)
    tar = 'log_diff'
    
    for i, pred in enumerate(syndromes):

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
                            color=viz.b1, ax=ax)
        ax.set_xlim([xmin, xmax]) # for the regression predictor 
        sns.regplot(x=pred, y=tar, data=data, truncate=False,
                        color=[.2, .2, .2], scatter=False, ax=ax)
        ax.set_ylabel(tar)
        ax.set_xlabel(pred)
        ax.set_xlim([xmin, xmax])
        #ax.set_ylim([-3., 3.])
        ax.set_box_aspect(1)


    plt.tight_layout()
    plt.savefig(f'{path}/figures/Fig0_LR_effect2.png', dpi=300)


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
                        palette=viz.RedPairs, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylim([-5, 5.8])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'PAT'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path}/figures/Fig1_HC-PAT-policies.png', dpi=300)

def STA_VOL_policy():

    tar    = ['l1', 'l2', 'l3']

    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data = data.groupby(by=['sub_id', 'feedback_type', 'b_type']).mean().reset_index()
    
    nr, nc = 1, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharey=True, sharex=True)
    for_title = t_test(data, 'b_type=="sta"', 'b_type=="vol"', tar=tar)
    for idx in range(nc):
        ax  = axs[idx]
        sns.boxplot(x='b_type', y=f'{tar[idx]}', data=data, width=.65,
                        palette=viz.BluePairs, ax=ax)
        ax.set_xlim([-.8, 1.8])
        ax.set_ylim([-5, 5.8])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Stable', 'Volatile'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_box_aspect(1)

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path}/figures/Fig4_STA-VOL-policies.png', dpi=300)

def Policy_Rew():

    tar    = ['l1', 'l2', 'l3']

    data = build_pivot_table('map', min_q=.01, max_q=.99)
    data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
    data['rew'] = data['rew'].apply(lambda x: x*100)
    data['rawRew'] = data['rawRew'].apply(lambda x: x*100)
    xmin, xmax = -4.9, 4.9 

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
        print(res.summary())
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

    fname = f'{path}/simulations/exp1data/MixPol/simsubj-exp1data-sta_first.csv'
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

def Block_Group_effect():

    tar = ['l1', 'l2', 'l3']
   
    nr, nc = 2, len(tar)
    fig, axs = plt.subplots(nr, nc, figsize=(nc*4, nr*4), sharey=True, sharex=True)
       
    for j in range(2):
        data = build_pivot_table('map', min_q=.01, max_q=.99)
        data['is_PAT'] = data['group'].apply(lambda x: x!='HC')
        sel_data = data.query(f'is_PAT=={j}').groupby(by=['sub_id', 'b_type', 'feedback_type']).mean().reset_index()
        t_test(sel_data, 'b_type=="sta"', 'b_type=="vol"', tar=tar)
        for idx in range(nc):
            ax  = axs[j, idx]
            sns.boxplot(x='b_type', y=f'{tar[idx]}', data=sel_data, width=.65,
                            palette=viz.BluePairs, ax=ax)
            ax.set_xlim([-.8, 1.8])
            ax.set_ylim([-5, 5.8])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Stable', 'Volatile'])
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_box_aspect(1)
        
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{path}/figures/Fig4_block-group-policies.png', dpi=300)

if __name__ == '__main__':

    #quantTable()
    #viz_Human()
    #LR_effect1()
    #LR_effect2()
    #viz_PiReward()
    #HC_PAT_policy()
    STA_VOL_policy()
    #Policy_Rew()
    #pred_biFactor()
    #pi_effect()
    #Block_Group_effect()